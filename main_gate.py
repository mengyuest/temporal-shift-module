import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import multiprocessing

import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder
from opts import parser

from tensorboardX import SummaryWriter
from ops.my_logger import Logger

from tools.net_flops_table import get_gflops_params, feat_dim_dict

#distributed
import platform
import torch.multiprocessing as mp
import torch.distributed as dist

# TODO(yue)
import numpy as np
import common
from os.path import join as ospj


def load_to_sd(model_path):
    if ".pth" in model_path:
        if os.path.exists(common.PRETRAIN_PATH + "/" + model_path):
            return torch.load(common.PRETRAIN_PATH + "/" + model_path)['state_dict']
        elif os.path.exists(common.EXPS_PATH + "/" + model_path):
            return torch.load(common.EXPS_PATH + "/" + model_path)['state_dict']
        else:
            exit("Cannot find model, exit")
    else:
        return {}


def main():
    args = parser.parse_args()

    #TODO(distributed)
    if args.hostfile != '':
        curr_node_name = platform.node().split(".")[0]
        with open(args.hostfile) as f:
            nodes = [x.strip() for x in f.readlines() if x.strip() != '']
            master_node = nodes[0].split(" ")[0]
        for idx, node in enumerate(nodes):
            if curr_node_name in node:
                args.rank = idx
                break
        args.world_size = len(nodes)
        args.dist_url = "tcp://{}:10598".format(master_node)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = len(args.gpus)

    test_mode = (args.test_from != "")

    logger = Logger()
    sys.stdout = logger

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, test_mode, logger))
    else:
        # Simply call main_worker function
        main_worker(args.gpus, ngpus_per_node, args, test_mode, logger)


def main_worker(gpu_i, ngpus_per_node, args, test_mode, logger):

    args.gpus = gpu_i
    if args.multiprocessing_distributed:
        node_seed_offset = 10086 * args.rank
    else:
        node_seed_offset = 0
    set_random_seed(node_seed_offset + args.random_seed, args)

    args.num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset)

    # TODO(distributed)
    if args.multiprocessing_distributed:
        print("Use GPU: {} for training".format(args.gpus))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the uniform rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpus
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = TSN_Gate(args=args)
    base_model_gflops, gflops_list = init_gflops_table(model, args)

    if args.no_optim:
        policies = [{'params': model.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "parameters"}]
    else:
        policies = model.get_optim_policies()

    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # TODO(distributed)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.multiprocessing_distributed:
            torch.cuda.set_device(args.gpus)
            model.cuda(args.gpus)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # the batch size should be divided by number of nodes as well
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(args.workers / ngpus_per_node)

            if args.sync_bn:
                process_group = torch.distributed.new_group(list(range(args.world_size)))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus], find_unused_parameters=True)
        else:
            exit("We don't do world_size>1 and distributed. We also don't do gpus=None")
            # model.cuda()
            # # DistributedDataParallel will divide and allocate batch_size to all
            # # available GPUs if device_ids are not set
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpus, find_unused_parameters=True)
    # elif args.multiprocessing_distributed:
    #     torch.cuda.set_device(args.gpus)
    #     model = model.cuda(args.gpus)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
        args.rank = 0

    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    handle_frozen_things_in(model, args)

    if args.resume:
        if os.path.isfile(args.resume):
            # TODO s
            if args.rank==0:
                print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.rank == 0:
                print(("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])))
        else:
            if args.rank == 0:
                print(("=> no checkpoint found at '{}'".format(args.resume)))

    # TODO(yue) loading pretrained weights
    elif test_mode or args.base_pretrained_from != "":
        if test_mode:
            the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
        else:
            the_model_path = args.base_pretrained_from
        model_dict = model.state_dict()
        sd = load_to_sd(the_model_path)

        old_to_new_pairs = []
        if args.downsample0_renaming:
            # TODO batchnorm
            for k in sd:
                # TODO downsample
                if "downsample0" in k:
                    # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                    old_to_new_pairs.append((k, k.replace("downsample0", "downsample.0")))
                elif "downsample1" in k:
                    # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                    old_to_new_pairs.append((k, k.replace("downsample1", "downsample.1")))

        if args.downsample_0_renaming:
            # TODO batchnorm
            for k in sd:
                # TODO downsample
                if "downsample.0" in k:
                    # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                    old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
                elif "downsample.1" in k:
                    # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                    old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

        for old_key, new_key in old_to_new_pairs:
            sd[new_key] = sd.pop(old_key)
        old_to_new_pairs = []

        if args.load_base_to_adaptive:
            for k in sd:
                if "base_model." in k:
                    old_to_new_pairs.append((k, k.replace("base_model.", "base_model_list.0.")))
                if "new_fc." in k:
                    old_to_new_pairs.append((k, k.replace("new_fc.", "new_fc_list.0.")))

        for old_key, new_key in old_to_new_pairs:
            sd[new_key] = sd.pop(old_key)

        old_to_new_pairs = []
        if args.shift and args.load_base_to_adaptive:
            for k in sd:
                if "conv1.net." in k:
                    old_to_new_pairs.append((k, k.replace("conv1.net.", "conv1.")))
        for old_key, new_key in old_to_new_pairs:
            sd[new_key] = sd.pop(old_key)

        del_keys = []
        if args.ignore_loading_gate_fc:
            del_keys += [k for k in sd if "gate_fc" in k]
        for k in del_keys:
            del sd[k]

        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(model, prefix, args)

    # define loss function (criterion) and optimizer
    if args.distributed:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpus)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    exp_full_path = setup_log_directory(logger, args.exp_header, test_mode, args)

    # TODO do simple eval here?

    if args.rank==0:
        if not test_mode:
            with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
                f.write(str(args))
        tf_writer = SummaryWriter(log_dir=exp_full_path)
    else:
        tf_writer = None

    # TODO(yue)
    if args.rank == 0:
        map_record, mmap_record, prec_record = get_recorders(3)
        best_train_usage_str = None
        best_val_usage_str = None
        val_gflops = -1

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(node_seed_offset + args.random_seed + epoch, args)
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list, args, tf_writer)

        torch.cuda.empty_cache()
        if args.distributed:
            dist.barrier()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, val_usage_str, val_gflops = validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, args, tf_writer)

            if args.distributed:
                dist.barrier()

            # remember best prec@1 and save checkpoint
            if args.rank == 0:
                map_record.update(mAP)
                mmap_record.update(mmAP)
                prec_record.update(prec1)

                best_by = {"map": map_record, "mmap": mmap_record, "acc": prec_record}
                if best_by[args.choose_best_by].is_current_best():
                    best_train_usage_str = train_usage_str if not args.skip_training else "(Eval Mode)"
                    best_val_usage_str = val_usage_str

                print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)' % (
                    map_record.best_val, map_record.best_at,
                    mmap_record.best_val, mmap_record.best_at,
                    prec_record.best_val, prec_record.best_at))

            if args.skip_training:
                break

            if args.rank==0:
                tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)
                saved_things = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': prec_record.best_val,
                    'best_mmap': mmap_record.best_val,
                    'best_map': map_record.best_val,
                }

                save_checkpoint(saved_things, mmap_record.is_current_best(), exp_full_path, "ckpt.bestmmap")
                save_checkpoint(saved_things, prec_record.is_current_best(), exp_full_path, "ckpt.best")
                save_checkpoint(saved_things, True, exp_full_path, "ckpt.latest")

                torch.cuda.empty_cache()

            if args.distributed:
                dist.barrier()


    # after fininshing all the epochs
    if args.rank == 0:
        if test_mode:
            os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                         "_mm_%.2f_a_%.2f_f_%.4f.txt" % (mmap_record.best_val, prec_record.best_val,
                                                                             val_gflops)))
        else:
            print("Best train usage:%s\nBest val usage:%s" % (best_train_usage_str, best_val_usage_str))


def set_random_seed(the_seed, args):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)


def init_gflops_table(model, args):
    if "cgnet" in args.arch:
        base_model_gflops = 1.8188 if "net18" in args.arch else 4.28
        params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)[1]
    else:
        base_model_gflops, params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)
    gflops_list = model.base_model_list[0].count_flops((1, 1, 3, args.reso_list[0], args.reso_list[0]))

    if args.rank==0:
        print("Network@%d (%.4f GFLOPS, %.4f M params) has %d blocks" % (args.reso_list[0], base_model_gflops, params, len(gflops_list)))
        for i, block in enumerate(gflops_list):
            print("block", i, ",".join(["%.4f GFLOPS" % (x / 1e9) for x in block]))
    return base_model_gflops, gflops_list


def compute_gflops_by_mask(mask_tensor_list, base_model_gflops, gflops_list, args):
    # TODO:   -> conv1 -> conv2 ->     // inside the block
    # TODO:    C0   -    C1   -    C2  // channels proc.
    # TODO:  C1 = s0 + s1 + s2         // 0-zero out / 1-history / 2-current     cheap <<< expensive
    # TODO: saving = s1/C1 * [FLOPS(conv1)] + s0/C1 * [FLOPS(conv1) + FLOPS(conv2)]
    # mask_cpu_list = [mask.detach().cpu() for mask in mask_tensor_list]
    # s0 = [torch.mean(mask_cpu[:, :, :, 0]).item() for mask_cpu in mask_cpu_list]
    # if args.gate_history:
    #     s1 = [torch.mean(mask_cpu[:, :, :, 1]).item() for mask_cpu in mask_cpu_list]
    # else:
    #     s1 = [0 for _ in mask_tensor_list]


    if "gate" in args.arch or "bate" in args.arch:
        if args.gate_no_skipping:
            s1 = [torch.mean(mask[:, :, :, 0]) for mask in mask_tensor_list]
            s1_savings = sum([s1[i] * (gflops_list[i][0] - gflops_list[i][-1]) for i in range(len(gflops_list))])
            real_gflops = base_model_gflops - s1_savings * 0
        else:
            s0 = [torch.mean(mask[:, :, :, 0]) for mask in mask_tensor_list]
            if args.gate_history:
                s1 = [torch.mean(mask[:, :, :, 1]) for mask in mask_tensor_list]
            else:
                s1 = [0 for _ in mask_tensor_list]

            if args.dense_in_block:
                if "net18" in args.arch or "net34" in args.arch:  # TODO BasicBlock
                    s0_savings = sum([s0[i * 2] * gflops_list[i][1] for i in range(len(gflops_list))])
                else:  # TODO BottleNeck
                    s0_savings_conv2 = sum([s0[i * 2] * gflops_list[i][1] for i in range(len(gflops_list))])
                    s0_savings_conv3 = sum([s0[i * 2 + 1] * gflops_list[i][2] for i in range(len(gflops_list))])
                    s0_savings = s0_savings_conv2 + s0_savings_conv3
            else:
                s0_savings = sum([s0[i] * gflops_list[i][1] for i in range(len(gflops_list))])
            s1_savings = 0
            # real_gflops = base_model_gflops - s0_savings / 1e9
            real_gflops = base_model_gflops - (s0_savings + s1_savings) / 1e9
    else:
        # s0 for sparsity savings
        # s1 for history
        # print(mask_tensor_list)
        # for mask in mask_tensor_list:
        #     print(torch.sum(mask[:, :, 1]), torch.sum(mask[:, :, 0]), torch.sum(mask[:, :, 1]) / torch.sum(mask[:, :, 0]))
        s0 = [1 - 1.0 * torch.sum(mask[:, :, 1]) / torch.sum(mask[:, :, 0]) for mask in mask_tensor_list]

        if args.dense_in_block:
            savings0 = sum([s0[i*2] * gflops_list[i][0] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
            savings1 = sum([s0[i*2+1] * gflops_list[i][1] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
            savings = savings0 + savings1
        else:
            savings = sum([s0[i] * gflops_list[i][0] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])

        real_gflops = base_model_gflops - savings / 1e9

    return real_gflops


def print_mask_statistics(mask_tensor_list, args):
    if "cgnet" in args.arch:
        cnt_ = [torch.sum(x, dim=[0, 1]) for x in mask_tensor_list]
        cnt_out = sum([x[0] for x in cnt_])
        cnt_full = sum([x[1] for x in cnt_])
        print("Overall sparsity: %.4f"%(1-1.0*cnt_full/cnt_out))
        print("Full:  ", " ".join(["%7.4f" % (1.0*x[1]/1e9) for x in cnt_]))
        print("Total: ", " ".join(["%7.4f" % (1.0*x[0]/1e9) for x in cnt_]))
        print("Ratio: ", " ".join(["%7.4f" % (1-1.0*x[1]/x[0]) for x in cnt_]))
        print()
    else:
        # overall
        # t=overall, 0, mid, end
        #     1. sparsity (layerwise)
        #     2. variance (layerwise)
        if args.gate_history:
            if args.gate_no_skipping:
                dim_str = ["hist", "curr"]
            else:
                dim_str = ["save", "hist", "curr"]
        else:
            dim_str = ["save", "curr"]

        for b_i in [None]: #[None, 0]:
            print("For example(b=0):" if b_i == 0 else "Overall:")
            b_start = b_i if b_i is not None else 0
            b_end = b_i + 1 if b_i is not None else mask_tensor_list[0].shape[0]

            for t_i in [None]: #[None, num_segments // 2]:  # "[None, 0, num_segments // 2, num_segments - 1]:
                t_start = t_i if t_i is not None else 0
                t_end = t_i + 1 if t_i is not None else mask_tensor_list[0].shape[1]
                for dim_i in range(len(dim_str)):
                    s_list = []
                    d_list = []
                    p_list = []

                    layer_i_list = [iii for iii in range(len(mask_tensor_list))]
                    cap_length = 8
                    if len(mask_tensor_list) > cap_length:
                        layer_i_list = [int(iii) for iii in np.linspace(0, len(mask_tensor_list)-1, cap_length, endpoint=True)]

                    for layer_i in layer_i_list:
                        s_list.append(torch.mean(mask_tensor_list[layer_i][b_start:b_end, t_start:t_end, :, dim_i]))
                        d_list.append(
                            torch.std(torch.mean(mask_tensor_list[layer_i][b_start:b_end, t_start:t_end, :, dim_i], dim=-1),
                                      unbiased=not (b_i == 0)))
                        # TODO channel-wise fire percentage for instances
                        # TODO this can be a channel percentage histogram, ranged from 0~1, where we only count five buckets
                        # TODO (0.00~0.20) (0.20~0.40) (0.40~0.60) (0.60~0.80) (0.80~1.00)
                        percentage = torch.mean(mask_tensor_list[layer_i][b_start:b_end, t_start:t_end, :, dim_i],
                                                dim=[0, 1])
                        p_list.append(torch.histc(percentage, bins=5, min=0, max=1) / percentage.shape[0])
                    t = "%3d" % t_i if t_i is not None else "all"
                    print("(t=%s, %s)usage: " % (t, dim_str[dim_i]),
                          "  ".join(["%.4f(%.4f) " % (s, d) for s, d in zip(s_list, d_list)]))
                    print("                  ",
                          " ".join(["(" + (",".join(["%02d" % (min(99, x * 100)) for x in p])) + ")" for p in p_list]))
            print()



        if args.dense_in_block:
            stat_skip0 = 0
            stat_reuse0 = 0
            stat_keep0 = 0
            stat_skip1 = 0
            stat_reuse1 = 0
            stat_keep1 = 0
            for layer_i, mask in enumerate(mask_tensor_list):
                if layer_i % 2 == 0:
                    if not args.gate_no_skipping:
                        stat_skip0 += torch.sum(mask[:, :, :, 0]).item()
                    if args.gate_history:
                        stat_reuse0 += torch.sum(mask[:, :, :, -2]).item()
                    stat_keep0 += torch.sum(mask[:, :, :, -1]).item()
                else:
                    if not args.gate_no_skipping:
                        stat_skip1 += torch.sum(mask[:, :, :, 0]).item()
                    if args.gate_history:
                        stat_reuse1 += torch.sum(mask[:, :, :, -2]).item()
                    stat_keep1 += torch.sum(mask[:, :, :, -1]).item()
            stat_total0 = stat_skip0 + stat_reuse0 + stat_keep0
            stat_total1 = stat_skip1 + stat_reuse1 + stat_keep1
            print("(=0)\nskip : %.4f  reuse: %.4f  keep : %.4f"
                  % (stat_skip0 / stat_total0, stat_reuse0 / stat_total0, stat_keep0 / stat_total0))

            print("(=1)\nskip : %.4f  reuse: %.4f  keep : %.4f"
                  % (stat_skip1 / stat_total1, stat_reuse1 / stat_total1, stat_keep1 / stat_total1))

        stat_skip = 0
        stat_reuse = 0
        stat_keep = 0
        for mask in mask_tensor_list:
            if not args.gate_no_skipping:
                stat_skip += torch.sum(mask[:, :, :, 0]).item()
            if args.gate_history:
                stat_reuse += torch.sum(mask[:, :, :, -2]).item()
            stat_keep += torch.sum(mask[:, :, :, -1]).item()

        stat_total = stat_skip + stat_reuse + stat_keep
        print("(overall)\nskip : %.4f  reuse: %.4f  keep : %.4f\n"
              % (stat_skip/stat_total, stat_reuse/stat_total, stat_keep/stat_total))



def reverse_onehot(a):
    try:
        return np.array([np.where(r > 0.5)[0][0] for r in a])
    except Exception as e:
        print("error stack:", e)
        print(a)
        for i, r in enumerate(a):
            print(i, r)
        return None


def compute_losses(criterion, prediction, target, mask_stack_list, gflops_tensor, epoch_i, model, args):
    # linear efficiency loss scheduling
    if args.gate_linear_phase > 0:
        factor = 1. / args.gate_linear_phase * min(args.gate_linear_phase, epoch_i)
    else:
        factor = 1.

    # accuracy loss
    acc_loss = criterion(prediction, target)

    # gflops loss
    if args.gate_gflops_loss_weight > 0:
        gflops_loss = torch.abs(gflops_tensor - args.gate_gflops_bias) * args.gate_gflops_loss_weight * factor
    else:
        gflops_loss = 0 * acc_loss

    # sparsity loss
    if args.gate_norm_loss_weight > 0:
        choice_dim = 3 if (args.gate_history and not args.gate_no_skipping) else 2
        mask_norm = torch.stack(
            [torch.norm(mask, dim=[0, 1, 2], p=args.gate_norm) / (mask.numel() / choice_dim) ** (1 / args.gate_norm)
             for mask in mask_stack_list], dim=0).mean(dim=0)
        if args.gate_history and args.gate_no_skipping:
            skip_mask_loss = acc_loss * 0
        else:
            skip_mask_loss = (1 - mask_norm[0]) * args.gate_norm_loss_factors[0] * args.gate_norm_loss_weight * factor
        if args.gate_history:
            if args.gate_no_skipping:
                hist_mask_loss = (1 - mask_norm[0]) * args.gate_norm_loss_factors[0] * args.gate_norm_loss_weight * factor
            else:
                hist_mask_loss = (1 - mask_norm[1]) * args.gate_norm_loss_factors[1] * args.gate_norm_loss_weight * factor
        else:
            hist_mask_loss = acc_loss * 0
        curr_mask_loss = mask_norm[-1] * args.gate_norm_loss_factors[-1] * args.gate_norm_loss_weight * factor
    else:
        skip_mask_loss = acc_loss * 0
        hist_mask_loss = acc_loss * 0
        curr_mask_loss = acc_loss * 0

    # threshold loss for cgnet
    thres_loss = acc_loss * 0
    if "cgnet" in args.arch:
        for name, param in model.named_parameters():
            if 'threshold' in name:
                # print(param)
                thres_loss += args.threshold_loss_weight * torch.sum((param-args.gtarget) ** 2)

    loss = acc_loss + gflops_loss + skip_mask_loss + hist_mask_loss + curr_mask_loss + thres_loss
    return {
        "loss": loss,
        "acc_loss": acc_loss,
        "eff_loss": loss - acc_loss,
        "gflops_loss": gflops_loss,
        "skip_mask_loss": skip_mask_loss,
        "hist_mask_loss": hist_mask_loss,
        "curr_mask_loss": curr_mask_loss,
        "thres_loss": thres_loss,
    }


def elastic_list_print(l, limit=8):
    if isinstance(l, str):
        return l
    limit = min(limit, len(l))
    l_output = "[%s," % (",".join([str(x) for x in l[:limit // 2]]))
    if l.shape[0] > limit:
        l_output += "..."
    l_output += "%s]" % (",".join([str(x) for x in l[-limit // 2:]]))
    return l_output


def compute_exp_decay_tau(epoch, args):
    return args.init_tau * np.exp(args.exp_decay_factor * epoch)


def get_policy_usage_str(gflops):
    return "Equivalent GFLOPS: %.4f" % (gflops.item())


def get_current_temperature(num_epoch, args):
    if args.exp_decay:
        tau = compute_exp_decay_tau(num_epoch, args)
    else:
        tau = args.init_tau
    return tau


def get_recorders(number):
    return [Recorder() for _ in range(number)]


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list, args, tf_writer):
    batch_time, data_time, top1, top5 = get_average_meters(4)
    losses_dict = {}
    mask_stack_list_list = [[] for _ in gflops_list] + [[] for _ in gflops_list] if args.dense_in_block else [[] for _ in gflops_list]
    tau = get_current_temperature(epoch, args)

    # switch to train mode
    model.module.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    if args.rank==0:
        print("#%s# lr:%.4f\ttau:%.4f" % (args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau))

    if dist.is_initialized():
        train_loader.sampler.set_epoch(epoch)

    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)

        # input and target
        batchsize = input_tuple[0].size(0)
        if args.distributed:
            input_var_list = [torch.autograd.Variable(input_item).cuda(args.gpus, non_blocking=True) for input_item in input_tuple[:-1]]
            target = input_tuple[-1].cuda(args.gpus, non_blocking=True)
        else:
            input_var_list = [torch.autograd.Variable(input_item).cuda(non_blocking=True) for input_item in
                              input_tuple[:-1]]
            target = input_tuple[-1].cuda(non_blocking=True)

        target_var = torch.autograd.Variable(target)

        # model forward function & measure losses and accuracy
        output, mask_stack_list, feat_out, base_outs = \
            model(input=input_var_list, tau=tau, is_training=True, curr_step=epoch * len(train_loader) + i)

        gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops, gflops_list, args)

        loss_dict = compute_losses(criterion, output, target_var[:, 0], mask_stack_list, gflops_tensor, epoch, model, args)
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(prec1)
            dist.all_reduce(prec5)
            prec1 /= world_size
            prec5 /= world_size

        # record losses and accuracy
        if len(losses_dict)==0:
            losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
        for loss_name in loss_dict:
            losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
        top1.update(prec1.item(), batchsize)
        top5.update(prec5.item(), batchsize)

        # compute gradient and do SGD step
        loss_dict["loss"].backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        optimizer.zero_grad()

        # gather masks
        for layer_i, mask_stack in enumerate(mask_stack_list):
            mask_stack_list_list[layer_i].append(mask_stack.detach().cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if args.rank==0 and i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                            'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss{loss.val:.4f}({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses_dict["loss"], top1=top1, top5=top5))  # TODO

            for loss_name in losses_dict:
                if loss_name == "loss":
                    continue
                print_output += ' {header:s} {loss.val:.3f}({loss.avg:.3f})'.\
                    format(header=loss_name[0], loss=losses_dict[loss_name])
            print(print_output)

    for layer_i in range(len(mask_stack_list_list)):
        mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
    batch_gflops = compute_gflops_by_mask(mask_stack_list_list, base_model_gflops, gflops_list, args)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(batch_gflops)
        batch_gflops /= world_size

    if args.rank==0:
        usage_str = get_policy_usage_str(batch_gflops)
        print(usage_str)
        if args.print_statistics:
            print_mask_statistics(mask_stack_list_list, args)

        if tf_writer is not None:
            tf_writer.add_scalar('loss/train', losses_dict["loss"].avg, epoch)
            tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    else:
        usage_str = "Empty (non-master nodes)"
    return usage_str


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, args, tf_writer=None):
    batch_time, top1, top5 = get_average_meters(3)
    # TODO(yue)
    all_results = []
    all_targets = []

    tau = get_current_temperature(epoch, args)

    # mask_stack_list_list = [[] for _ in gflops_list]
    mask_stack_list_list = [[] for _ in gflops_list] + [[] for _ in gflops_list] if args.dense_in_block else [[] for _
                                                                                                               in
                                                                                                               gflops_list]

    losses_dict={}

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            # input and target
            batchsize = input_tuple[0].size(0)
            if args.distributed:
                input_tuple = [x.cuda(args.gpus, non_blocking=True) for x in input_tuple]
            else:
                input_tuple = [x.cuda(non_blocking=True) for x in input_tuple]
            target = input_tuple[-1]
            # target = input_tuple[-1].cuda(args.gpus, non_blocking=True)

            # model forward function
            output, mask_stack_list, feat_out, base_outs = \
                model(input=input_tuple[:-1], tau=tau, is_training=False, curr_step=0)

            gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops, gflops_list, args)

            # measure losses, accuracy and predictions
            loss_dict = compute_losses(criterion, output, target[:, 0], mask_stack_list, gflops_tensor, epoch, model, args)

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size


            all_results.append(output)
            all_targets.append(target)

            # record loss and accuracy
            if len(losses_dict) == 0:
                losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
            for loss_name in loss_dict:
                losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
            top1.update(prec1.item(), batchsize)
            top5.update(prec5.item(), batchsize)

            # gather masks
            for layer_i, mask_stack in enumerate(mask_stack_list):
                mask_stack_list_list[layer_i].append(mask_stack.detach().cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank==0 and i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.
                                format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses_dict["loss"], top1=top1, top5=top5))

                for loss_name in losses_dict:
                    if loss_name == "loss":
                        continue
                    print_output += ' {header:s} {loss.val:.3f}({loss.avg:.3f})'. \
                        format(header=loss_name[0], loss=losses_dict[loss_name])
                print(print_output)

    for layer_i in range(len(mask_stack_list_list)):
        mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
    batch_gflops = compute_gflops_by_mask(mask_stack_list_list, base_model_gflops, gflops_list, args)

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue)  multi-label mAP



    if dist.is_initialized():
        mAP_tensor = torch.tensor(mAP)
        mmAP_tensor = torch.tensor(mmAP)

        world_size = dist.get_world_size()
        dist.all_reduce(batch_gflops)
        dist.all_reduce(mAP_tensor)
        dist.all_reduce(mmAP_tensor)
        batch_gflops /= world_size
        mAP_tensor /= world_size
        mmAP_tensor /= world_size
        mAP = mAP_tensor.item()
        mmAP = mmAP_tensor.item()

    if args.rank==0:
        print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses_dict["loss"]))
        usage_str = get_policy_usage_str(batch_gflops)
        print(usage_str)
        if args.print_statistics:
            print_mask_statistics(mask_stack_list_list, args)

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses_dict["loss"].avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
    else:
        usage_str = "Empty: non-master node"
    return mAP, mmAP, top1.avg, usage_str, batch_gflops


def save_checkpoint(state, is_best, exp_full_path, decorator):
    if is_best:
        torch.save(state, '%s/models/%s.pth.tar' % (exp_full_path, decorator))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def setup_log_directory(logger, exp_header, test_mode, args):
    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    if test_mode:
        exp_full_path = ospj(common.EXPS_PATH, args.test_from)
    else:
        exp_full_path = ospj(common.EXPS_PATH, exp_full_name)
        if args.rank == 0:
            os.makedirs(exp_full_path)
            os.makedirs(ospj(exp_full_path, "models"))
    if args.rank == 0:
        logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path

def build_dataflow(dataset, is_train, batch_size, workers, is_distributed, not_pin_memory):
    workers = min(workers, multiprocessing.cpu_count())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    shuffle = False
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=not not_pin_memory, sampler=sampler,
                                              drop_last=is_train)
    return data_loader


def get_data_loaders(model, prefix, args):
    if args.rank == 0:
        print("%s: %d classes" % (args.dataset, args.num_class))

    crop_size = model.module.crop_size
    scale_size = model.module.scale_size
    input_mean = model.module.input_mean
    input_std = model.module.input_std

    train_augmentation = model.module.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    normalize = GroupNormalize(input_mean, input_std)


    train_dataset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   filelist_suffix=args.filelist_suffix,
                   folder_suffix=args.folder_suffix,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta,
                   rank=args.rank)

    val_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   filelist_suffix=args.filelist_suffix,
                   folder_suffix=args.folder_suffix,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta,
                   rank=args.rank)

    # train_loader = torch.utils.data.DataLoader(
    #     ,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=workers, pin_memory=not args.not_pin_memory,
    #     drop_last=True)  # prevent something not % n_GPU
    #
    # val_loader = torch.utils.data.DataLoader(
    #     ,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=not args.not_pin_memory)

    train_loader = build_dataflow(train_dataset, True, args.batch_size, args.workers, args.distributed, args.not_pin_memory)
    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.distributed, args.not_pin_memory)

    return train_loader, val_loader


def handle_frozen_things_in(model, args):
    # TODO(yue) freeze some params in the policy + lstm layers
    if args.freeze_policy:
        for name, param in model.module.named_parameters():
            if "lite_fc" in name or "lite_backbone" in name or "rnn" in name or "linear" in name:
                param.requires_grad = False

    if args.freeze_backbone:
        for name, param in model.module.named_parameters():
            if "base_model" in name:
                param.requires_grad = False
    if len(args.frozen_list) > 0:
        for name, param in model.module.named_parameters():
            for keyword in args.frozen_list:
                if keyword[0] == "*":
                    if keyword[-1] == "*":  # TODO middle
                        if keyword[1:-1] in name:
                            param.requires_grad = False
                            if args.rank==0:
                                print(keyword, "->", name, "frozen")
                    else:  # TODO suffix
                        if name.endswith(keyword[1:]):
                            param.requires_grad = False
                            if args.rank == 0:
                                print(keyword, "->", name, "frozen")
                elif keyword[-1] == "*":  # TODO prefix
                    if name.startswith(keyword[:-1]):
                        param.requires_grad = False
                        if args.rank == 0:
                            print(keyword, "->", name, "frozen")
                else:  # TODO exact word
                    if name == keyword:
                        param.requires_grad = False
                        if args.rank == 0:
                            print(keyword, "->", name, "frozen")
        if args.rank == 0:
            print("=" * 80)
            for name, param in model.module.named_parameters():
                print(param.requires_grad, "\t", name)

    if len(args.frozen_layers) > 0:
        for layer_idx in args.frozen_layers:
            for name, param in model.module.named_parameters():
                if layer_idx == 0:
                    if "list.0.conv1" in name:
                        param.requires_grad = False
                        if args.rank == 0:
                            print(layer_idx, "->", name, "frozen")
                else:
                    if "list.0.layer%d" % layer_idx in name and ("conv" in name or "downsample.0" in name):
                        param.requires_grad = False
                        if args.rank == 0:
                            print(layer_idx, "->", name, "frozen")
            if args.freeze_corr_bn:
                for km in model.named_modules():
                    k, m = km
                    if layer_idx == 0:
                        if "bn1" in k and "layer" not in k and isinstance(m, nn.BatchNorm2d):  # TODO(yue)
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                            if args.rank == 0:
                                print(layer_idx, "->", k, "frozen batchnorm")
                    else:
                        if "layer%d" % (layer_idx) in k and isinstance(m, nn.BatchNorm2d):  # TODO(yue)
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                            if args.rank == 0:
                                print(layer_idx, "->", k, "frozen batchnorm")
        if args.rank == 0:
            print("=" * 80)
            for name, param in model.module.named_parameters():
                print(param.requires_grad, "\t", name)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("Finished in %.4f seconds\n" % (time.time() - t0))
