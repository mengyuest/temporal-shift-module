import warnings

warnings.filterwarnings("ignore")

import os
from os.path import expanduser
import sys
import time
import multiprocessing

import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as f
from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.models_ada import TSN_Ada
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, verb_noun_accuracy, get_marginal_output
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
from shutil import copyfile
import shutil

def main():
    args = parser.parse_args()
    common.set_manual_data_path(args.data_path, args.exps_path)

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
    ngpus_per_node = torch.cuda.device_count() #len(args.gpus)

    test_mode = (args.test_from != "")
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, test_mode))
    else:
        main_worker(args.gpus, ngpus_per_node, args, test_mode)


def main_worker(gpu_i, ngpus_per_node, args, test_mode):
    args.gpus = gpu_i
    if args.multiprocessing_distributed or args.distributed:
        node_seed_offset = 10086 * args.rank
    else:
        node_seed_offset = 0
    set_random_seed(node_seed_offset + args.random_seed, args)

    args.num_class, args.train_list, args.val_list, args.root_path, prefix = \
        dataset_config.return_dataset(args.dataset,
                                      args.data_path)  # TODO this is only used if manually set

    # TODO(distributed)
    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the uniform rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpus
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        logger = Logger()
        sys.stdout = logger

    if args.ada_reso_skip:
        model = TSN_Gate(args=args)
    else:
        model = TSN_Ada(args=args)
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
        if args.gpus is not None and not isinstance(args.gpus, list):
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
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    # elif args.gpus is not None:
    #     torch.cuda.set_device(args.gpus)
    #     model = model.cuda(args.gpus)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
        args.rank = 0

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
    elif test_mode or args.base_pretrained_from != "" or args.use_tsmk8 or args.use_segk8 or args.use_tsmk16:
        if args.use_segk8:
            the_model_path = expanduser(
                "~/.cache/torch/checkpoints/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth")
        elif args.use_tsmk8:
            the_model_path = expanduser(
                "~/.cache/torch/checkpoints/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth")
        elif args.use_tsmk16:
            the_model_path = expanduser(
                "~/.cache/torch/checkpoints/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth")
        else:
            the_model_path = args.base_pretrained_from
            if test_mode:
                the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
            the_model_path = common.EXPS_PATH + "/" + the_model_path

        sd = torch.load(the_model_path)['state_dict']
        sd = take_care_of_pretraining(sd, args)

        model_dict = model.state_dict()
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(model, prefix, args)
    # define loss function (criterion) and optimizer
    # if args.gpus is not None: #args.distributed
    #     criterion = torch.nn.CrossEntropyLoss().cuda(args.gpus)
    # else:
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.rank==0:
        exp_full_path = setup_log_directory(args.exp_header, test_mode, args, logger)
        # TODO stat runtime info
        import socket
        import getpass
        print("%s@%s started the experiment at %s"%(getpass.getuser(), socket.gethostname(), logger._timestr))

        if not test_mode:
            with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
                f.write(str(args))
        tf_writer = SummaryWriter(log_dir=exp_full_path)
    else:
        tf_writer = None

    # TODO(yue)
    if args.rank == 0:
        map_record, mmap_record, prec_record, prec5_record = get_recorders(4)
        if args.dataset == "epic":
            verb_prec1_record, verb_prec5_record, noun_prec1_record, noun_prec5_record = get_recorders(4)
        best_train_usage_str = None
        best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(node_seed_offset + args.random_seed + epoch, args)
            adjust_learning_rate(optimizer, epoch, -1, -1, args.lr_type, args.lr_steps, args)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list, args, tf_writer)

        torch.cuda.empty_cache()
        if args.distributed:
            dist.barrier()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, prec5, val_usage_str, epic_precs = \
                validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, exp_full_path, args, tf_writer)

            if args.distributed:
                dist.barrier()

            # remember best prec@1 and save checkpoint
            if args.rank == 0:
                map_record.update(mAP)
                mmap_record.update(mmAP)
                prec_record.update(prec1)
                prec5_record.update(prec5)
                if args.dataset == "epic":
                    verb_prec1_record.update(epic_precs[0])
                    verb_prec5_record.update(epic_precs[1])
                    noun_prec1_record.update(epic_precs[2])
                    noun_prec5_record.update(epic_precs[3])

                best_by = {"map": map_record, "mmap": mmap_record, "acc": prec_record}
                if best_by[args.choose_best_by].is_current_best():
                    best_train_usage_str = train_usage_str if not args.skip_training else "(Eval Mode)"
                    best_val_usage_str = val_usage_str

                epic_str = ""
                if args.dataset == "epic":
                    epic_str = "V@1:%.3f V@5:%.3f N@1:%.3f N@5:%.3f" % (
                        verb_prec1_record.best_val, verb_prec5_record.best_val,
                        noun_prec1_record.best_val, noun_prec5_record.best_val
                    )

                print('Best mAP: %.3f (epoch=%d)\tBest mmAP: %.3f(epoch=%d)\tBest Prec@1: %.3f (epoch=%d) w. Prec@5: %.3f %s' % (
                    map_record.best_val, map_record.best_at,
                    mmap_record.best_val, mmap_record.best_at,
                    prec_record.best_val, prec_record.best_at,
                    prec5_record.at(prec_record.best_at), epic_str
                ))

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
                    'prec5_at_best_prec1': prec5_record.at(prec_record.best_at),
                    'best_mmap': mmap_record.best_val,
                    'best_map': map_record.best_val,
                }

                save_checkpoint(saved_things, prec_record.is_current_best(), False, exp_full_path, "ckpt.best")
                save_checkpoint(saved_things, True, False, exp_full_path, "ckpt.latest")

                if epoch in args.backup_epoch_list:
                    save_checkpoint(None, False, True, exp_full_path, str(epoch))

                torch.cuda.empty_cache()

            if args.distributed:
                dist.barrier()

    # after fininshing all the epochs
    if args.rank == 0:
        if test_mode:
            os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                         "_mm_%.2f_a_%.2f_f.txt" % (mmap_record.best_val, prec_record.best_val)))
        else:
            if args.ada_reso_skip:
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

    if args.ada_reso_skip:
        gflops_list = model.base_model.count_flops((1, 1, 3, args.reso_list[0], args.reso_list[0]))
        if args.rank==0:
            print("Network@%d (%.4f GFLOPS, %.4f M params) has %d blocks" % (args.reso_list[0], base_model_gflops, params, len(gflops_list)))
            for i, block in enumerate(gflops_list):
                print("block", i, ",".join(["%.4f GFLOPS" % (x / 1e9) for x in block]))
        return base_model_gflops, gflops_list
    else:
        if args.rank == 0:
            print("Network@%d (%.4f GFLOPS, %.4f M params)" % (args.reso_list[0], base_model_gflops, params))
        return base_model_gflops, None

def compute_gflops_by_mask(mask_tensor_list, base_model_gflops, gflops_list, args):
    # TODO:   -> conv1 -> conv2 ->     // inside the block
    # TODO:    C0   -    C1   -    C2  // channels proc.
    # TODO:  C1 = s0 + s1 + s2         // 0-zero out / 1-history / 2-current     cheap <<< expensive
    # TODO: saving = s1/C1 * [FLOPS(conv1)] + s0/C1 * [FLOPS(conv1) + FLOPS(conv2)]
    upperbound_gflops = base_model_gflops
    real_gflops = base_model_gflops

    if "bate" in args.arch:
        for m_i, mask in enumerate(mask_tensor_list):
            #compute precise GFLOPS
            upsave = torch.zeros_like(mask[:, :, :, 0]) # B*T*C*K->B*T*C
            for t in range(mask.shape[1]-1):
                if args.gate_history:
                    upsave[:, t] = (1 - mask[:, t, :, -1]) * (1 - mask[:, t + 1, :, -2])
                else:
                    upsave[:, t] = 1 - mask[:, t, :, -1] # since no reusing, as long as not keeping, save from upstream conv
            upsave[:, -1] = 1 - mask[:, t, :, -1]
            upsave = torch.mean(upsave)

            if args.gate_no_skipping: # downstream conv gflops' saving is from skippings
                downsave = upsave * 0
            else:
                downsave = torch.mean(mask[:, :, :, 0])

            conv_offset = 0
            real_count = 1.
            if args.dense_in_block:
                layer_i = m_i // 2  # because we have twice masks as the #(blocks)
                if m_i % 2 == 1:  # means we come to the second mask in the block
                    if "net50" in args.arch or "net101" in args.arch:  # because we have 3 convs in BottleNeck
                        conv_offset = 1
                    else:  # because we can't compute flops saving among blocks (due to residual op), so we skip this (as this is the case only in BasicBlock)
                        real_count = 0
            else:
                layer_i = m_i
            up_flops = gflops_list[layer_i][0 + conv_offset] / 1e9
            down_flops = gflops_list[layer_i][1 + conv_offset] * real_count / 1e9
            embed_conv_flops = gflops_list[layer_i][-1] * real_count / 1e9

            upperbound_gflops = upperbound_gflops - downsave * (down_flops - embed_conv_flops) # in worst case, we only compute saving from downstream conv
            real_gflops = real_gflops - upsave * up_flops - downsave * (down_flops - embed_conv_flops)

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
        upperbound_gflops = real_gflops

    return upperbound_gflops, real_gflops


def print_mask_statistics(mask_tensor_list, args):
    if "cgnet" in args.arch:
        cnt_ = [torch.sum(x, dim=[0, 1]) for x in mask_tensor_list] # sum up over t
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

        print("Overall:")

        normalized_tensor_list=[]
        for mask in mask_tensor_list:
            normalized_tensor_list.append(f.normalize(mask, dim=-1, p=1))

        for t_i in [None]: #[None, num_segments // 2]:  # "[None, 0, num_segments // 2, num_segments - 1]:
            t_start = t_i if t_i is not None else 0
            t_end = t_i + 1 if t_i is not None else mask_tensor_list[0].shape[0]
            for dim_i in range(len(dim_str)):
                s_list = []

                layer_i_list = [iii for iii in range(len(mask_tensor_list))]
                cap_length = 8
                if len(mask_tensor_list) > cap_length:
                    layer_i_list = [int(iii) for iii in np.linspace(0, len(mask_tensor_list)-1, cap_length, endpoint=True)]

                for layer_i in layer_i_list:
                    s_list.append(torch.mean(normalized_tensor_list[layer_i][t_start:t_end, :, dim_i]))
                    # d_list.append(
                    #     torch.std(torch.mean(mask_tensor_list[layer_i][t_start:t_end, :, dim_i], dim=-1), unbiased=True))
                    # TODO channel-wise fire percentage for instances
                    # TODO this can be a channel percentage histogram, ranged from 0~1, where we only count five buckets
                    # TODO (0.00~0.20) (0.20~0.40) (0.40~0.60) (0.60~0.80) (0.80~1.00)
                    # percentage = torch.mean(mask_tensor_list[layer_i][t_start:t_end, :, dim_i], dim=[0])
                    # p_list.append(torch.histc(percentage, bins=5, min=0, max=1) / percentage.shape[0])
                t = "%3d" % t_i if t_i is not None else "all"
                print("(t=%s, %s)usage: " % (t, dim_str[dim_i]),
                      "  ".join(["%.4f " % (s) for s in s_list]))
                # print("                  ",
                #       " ".join(["(" + (",".join(["%02d" % (min(99, x * 100)) for x in p])) + ")" for p in p_list]))
            print()

        # if args.dense_in_block:
        #     stat_skip0 = 0
        #     stat_reuse0 = 0
        #     stat_keep0 = 0
        #     stat_skip1 = 0
        #     stat_reuse1 = 0
        #     stat_keep1 = 0
        #     for layer_i, mask in enumerate(mask_tensor_list):
        #         if layer_i % 2 == 0:
        #             if not args.gate_no_skipping:
        #                 stat_skip0 += torch.sum(mask[:, :, :, 0]).item()
        #             if args.gate_history:
        #                 stat_reuse0 += torch.sum(mask[:, :, :, -2]).item()
        #             stat_keep0 += torch.sum(mask[:, :, :, -1]).item()
        #         else:
        #             if not args.gate_no_skipping:
        #                 stat_skip1 += torch.sum(mask[:, :, :, 0]).item()
        #             if args.gate_history:
        #                 stat_reuse1 += torch.sum(mask[:, :, :, -2]).item()
        #             stat_keep1 += torch.sum(mask[:, :, :, -1]).item()
        #     stat_total0 = stat_skip0 + stat_reuse0 + stat_keep0
        #     stat_total1 = stat_skip1 + stat_reuse1 + stat_keep1
        #     print("(=0)\nskip : %.4f  reuse: %.4f  keep : %.4f"
        #           % (stat_skip0 / stat_total0, stat_reuse0 / stat_total0, stat_keep0 / stat_total0))
        #
        #     print("(=1)\nskip : %.4f  reuse: %.4f  keep : %.4f"
        #           % (stat_skip1 / stat_total1, stat_reuse1 / stat_total1, stat_keep1 / stat_total1))

        # stat_skip = 0
        # stat_reuse = 0
        # stat_keep = 0
        # for mask in mask_tensor_list:
        #     if not args.gate_no_skipping:
        #         stat_skip += torch.sum(mask[:, :, 0]).item()
        #     if args.gate_history:
        #         stat_reuse += torch.sum(mask[:, :, -2]).item()
        #     stat_keep += torch.sum(mask[:, :, -1]).item()
        #
        # stat_total = stat_skip + stat_reuse + stat_keep

        stat_skip, stat_reuse, stat_keep, stat_total = get_mask_usage(mask_tensor_list, args)

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

def get_mask_usage(mask_tensor_list, args):
    stat_skip = 0
    stat_reuse = 0
    stat_keep = 0
    for mask in mask_tensor_list:
        if not args.gate_no_skipping:
            stat_skip += torch.sum(mask[:, :, 0])
        if args.gate_history:
            stat_reuse += torch.sum(mask[:, :, -2])
        stat_keep += torch.sum(mask[:, :, -1])

    stat_total = stat_skip + stat_reuse + stat_keep
    return stat_skip, stat_reuse, stat_keep, stat_total

def compute_epic_losses(criterion, prediction, target, a_v_m, a_n_m):
    v_pred = get_marginal_output(prediction, a_v_m, 125)
    n_pred = get_marginal_output(prediction, a_n_m, 352)
    v_acc_loss = criterion(v_pred, target[:, 1])
    n_acc_loss = criterion(n_pred, target[:, 2])
    acc_loss = v_acc_loss + n_acc_loss
    return v_acc_loss, n_acc_loss, acc_loss

def compute_losses(criterion, prediction, target, mask_stack_list, upb_gflops_tensor, real_gflops_tensor, epoch_i, model,
                   a_v_m, a_n_m, base_model_gflops, args):
    loss_dict={}
    if args.gflops_loss_type == "real":
        gflops_tensor = real_gflops_tensor
    else:
        gflops_tensor = upb_gflops_tensor

    # linear efficiency loss scheduling
    if args.gate_linear_phase > 0:
        factor = 1. / args.gate_linear_phase * min(args.gate_linear_phase, epoch_i)
    else:
        factor = 1.

    if epoch_i < args.gate_loss_starts_from:
        factor = 0.

    # accuracy loss
    if args.dataset == "epic":  # combined_verb/noun_losses
        v_acc_loss, n_acc_loss, acc_loss = compute_epic_losses(criterion, prediction, target, a_v_m, a_n_m)
        loss_dict["verb_loss"] = v_acc_loss
        loss_dict["noun_loss"] = n_acc_loss
    else:
        acc_loss = criterion(prediction, target[:, 0])
        loss_dict["acc_loss"] = acc_loss

    loss_dict["eff_loss"] = acc_loss * 0
    # gflops loss
    gflops_loss = acc_loss * 0
    if args.gate_gflops_loss_weight > 0 and epoch_i > args.eff_loss_after:
        if args.gflops_loss_norm == 1:
            gflops_loss = torch.abs(gflops_tensor - args.gate_gflops_bias) * args.gate_gflops_loss_weight * factor
        elif args.gflops_loss_norm == 2:
            gflops_loss = ((gflops_tensor/base_model_gflops - args.gate_gflops_threshold)**2) * args.gate_gflops_loss_weight * factor
        loss_dict["gflops_loss"] = gflops_loss
        loss_dict["eff_loss"] += gflops_loss

    # regularizer loss
    regu_loss = acc_loss * 0
    if args.keep_weight > 0 or args.reuse_weight>0 or args.skip_weight>0:
        stat_skip, stat_reuse, stat_keep, stat_total = get_mask_usage(mask_stack_list, args)
        regu_loss += args.skip_weight * (((stat_skip / stat_total - args.skip_ratio) / args.skip_ratio) ** 2)
        regu_loss += args.reuse_weight * (((stat_reuse / stat_total - args.reuse_ratio) / args.reuse_ratio) ** 2)
        regu_loss += args.keep_weight * (((stat_keep / stat_total - args.keep_ratio) / args.keep_ratio) ** 2)
        loss_dict["regu_loss"] = regu_loss
        loss_dict["eff_loss"] += regu_loss

    # threshold loss for cgnet
    thres_loss = acc_loss * 0
    if "cgnet" in args.arch:
        for name, param in model.named_parameters():
            if 'threshold' in name:
                # print(param)
                thres_loss += args.threshold_loss_weight * torch.sum((param-args.gtarget) ** 2)
        loss_dict["thres_loss"] = thres_loss
        loss_dict["eff_loss"] += thres_loss
    loss = acc_loss + gflops_loss + thres_loss
    loss_dict["loss"] = loss

    return loss_dict
    # if args.dataset == "epic":
    #     return {
    #         "loss": loss,
    #         "verb_loss": v_acc_loss,
    #         "noun_loss": n_acc_loss,
    #         "eff_loss": loss - acc_loss,
    #         "regu_loss": loss - acc_loss,
    #         "gflops_loss": gflops_loss,
    #         "thres_loss": thres_loss,
    #     }
    #
    # else:
    #     return {
    #         "loss": loss,
    #         "acc_loss": acc_loss,
    #         "eff_loss": loss - acc_loss,
    #         "gflops_loss": gflops_loss,
    #         "thres_loss": thres_loss,
    #     }

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


def get_policy_usage_str(upb_gflops, real_gflops):
    return "Equivalent GFLOPS: upb: %.4f   real: %.4f" % (upb_gflops.item(), real_gflops.item())


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
    if args.dataset=="epic":
        verb_top1, verb_top5, noun_top1, noun_top5 = get_average_meters(4)
    losses_dict = {}
    if args.ada_reso_skip:

        if "batenet" in args.arch:
            mask_stack_list_list = [0 for _ in gflops_list] + [0 for _ in gflops_list] if args.dense_in_block else [0 for
                                                                                                                  _ in
                                                                                                                  gflops_list]
        else:
            mask_stack_list_list = [[] for _ in gflops_list] + [[] for _ in gflops_list] if args.dense_in_block else [[] for _ in gflops_list]
        upb_batch_gflops_list=[]
        real_batch_gflops_list=[]

    tau = get_current_temperature(epoch, args)

    # switch to train mode
    model.module.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    if args.rank==0:
        print("#%s# lr:%.6f\ttau:%.4f" % (args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau))

    if dist.is_initialized():
        train_loader.sampler.set_epoch(epoch)

    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.warmup_epochs > 0:
            adjust_learning_rate(optimizer, epoch, len(train_loader), i, "linear", None, args)

        # input and target
        batchsize = input_tuple[0].size(0)
        # if args.gpus is not None:
        #     input_var_list = [torch.autograd.Variable(input_item).cuda(args.gpus, non_blocking=True) for input_item in input_tuple[:-1]]
        #     target = input_tuple[-1].cuda(args.gpus, non_blocking=True)
        # else:
        input_var_list = [torch.autograd.Variable(input_item).cuda(non_blocking=True) for input_item in
                          input_tuple[:-1]]
        target = input_tuple[-1].cuda(non_blocking=True)

        target_var = torch.autograd.Variable(target)

        # model forward function & measure losses and accuracy
        output, mask_stack_list, _, _ = \
            model(input=input_var_list, tau=tau, is_training=True, curr_step=epoch * len(train_loader) + i)

        if args.ada_reso_skip:
            # for m_i in range(len(mask_stack_list)):
            #     mask_stack_list[m_i] = torch.sum(mask_stack_list[m_i], dim=0)
            #     mask_stack_list[m_i] = f.normalize(mask_stack_list[m_i], dim=-1, p=1)
            upb_gflops_tensor, real_gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops, gflops_list, args)
            loss_dict = compute_losses(criterion, output, target_var, mask_stack_list,
                                       upb_gflops_tensor, real_gflops_tensor, epoch, model,
                                       train_loader.a_v_m, train_loader.a_n_m, base_model_gflops, args)
            upb_batch_gflops_list.append(upb_gflops_tensor.detach())
            real_batch_gflops_list.append(real_gflops_tensor.detach())
        else:
            if args.dataset == "epic":
                v_acc_loss, n_acc_loss, acc_loss = compute_epic_losses(criterion, output, target_var,
                                                                       train_loader.a_v_m, train_loader.a_n_m)
                loss_dict = {"loss": acc_loss, "verb_loss": v_acc_loss, "noun_loss": n_acc_loss}
            else:
                loss_dict = {"loss": criterion(output, target_var[:, 0])}
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

        if args.dataset == "epic":
            verb_prec1, verb_prec5, noun_prec1, noun_prec5 = verb_noun_accuracy(
                train_loader.a_v_m, train_loader.a_n_m, output.data, target, topk=(1, 5))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(prec1)
            dist.all_reduce(prec5)
            prec1 /= world_size
            prec5 /= world_size
            if args.dataset == "epic":
                dist.all_reduce(verb_prec1)
                dist.all_reduce(verb_prec5)
                dist.all_reduce(noun_prec1)
                dist.all_reduce(noun_prec5)
                verb_prec1 /= world_size
                verb_prec5 /= world_size
                noun_prec1 /= world_size
                noun_prec5 /= world_size

        # record losses and accuracy
        if len(losses_dict)==0:
            losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
        for loss_name in loss_dict:
            losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
        top1.update(prec1.item(), batchsize)
        top5.update(prec5.item(), batchsize)
        if args.dataset == "epic":
            verb_top1.update(verb_prec1.item(), batchsize)
            verb_top5.update(verb_prec5.item(), batchsize)
            noun_top1.update(noun_prec1.item(), batchsize)
            noun_top5.update(noun_prec5.item(), batchsize)

        # compute gradient and do SGD step
        loss_dict["loss"].backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        optimizer.zero_grad()

        # gather masks
        if args.ada_reso_skip:
            for layer_i, mask_stack in enumerate(mask_stack_list):
                if "batenet" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)
                else:  # TODO CGNet
                    mask_stack_list_list[layer_i].append(mask_stack.detach()) #TODO removed cpu()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if args.rank==0 and i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] lr {3:.6f} '
                            'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss{loss.val:.4f}({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), optimizer.param_groups[-1]['lr'] * 0.1, batch_time=batch_time,
                data_time=data_time, loss=losses_dict["loss"], top1=top1, top5=top5))  # TODO

            if args.dataset=="epic":
                print_output += "V@1({v1.avg:.3f}) ({v5.avg:.3f}) " \
                                "N@1({n1.avg:.3f}) ({n5.avg:.3f}) ".\
                    format(v1=verb_top1, v5=verb_top5, n1=noun_top1, n5=noun_top5)

            for loss_name in losses_dict:
                if loss_name == "loss" or "mask" in loss_name:
                    continue
                print_output += ' {header:s} ({loss.avg:.3f})'.\
                    format(header=loss_name[0], loss=losses_dict[loss_name])
            print(print_output)
    if args.ada_reso_skip:
        if "cgnet" in args.arch:
            for layer_i in range(len(mask_stack_list_list)):
                mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
        # upb_batch_gflops, real_batch_gflops = compute_gflops_by_mask(mask_stack_list_list, base_model_gflops, gflops_list, args)
        upb_batch_gflops= torch.mean(torch.stack(upb_batch_gflops_list))
        real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(upb_batch_gflops)
            dist.all_reduce(real_batch_gflops)
            upb_batch_gflops /= world_size
            real_batch_gflops /= world_size

    if args.rank == 0:
        if args.ada_reso_skip:
            usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
            print(usage_str)
            # if args.print_statistics:
            print_mask_statistics(mask_stack_list_list, args)
        else:
            usage_str = "Base Model"
        if tf_writer is not None:
            tf_writer.add_scalar('loss/train', losses_dict["loss"].avg, epoch)
            tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    else:
        usage_str = "Empty (non-master nodes)"
    return usage_str


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, exp_full_path, args, tf_writer=None):
    batch_time, top1, top5 = get_average_meters(3)
    if args.dataset=="epic":
        verb_top1, verb_top5, noun_top1, noun_top5 = get_average_meters(4)
    # TODO(yue)
    all_results = []
    all_targets = []

    if args.save_meta_gate:
        gate_meta_list = []
        mask_stat_list = []

    tau = get_current_temperature(epoch, args)

    if args.ada_reso_skip:
        if "batenet" in args.arch:
            mask_stack_list_list = [0 for _ in gflops_list] + [0 for _ in gflops_list] if args.dense_in_block else [0 for
                                                                                                                  _ in
                                                                                                                  gflops_list]
        else:
            mask_stack_list_list = [[] for _ in gflops_list] + [[] for _ in gflops_list] if args.dense_in_block else [[] for _ in gflops_list]
        upb_batch_gflops_list = []
        real_batch_gflops_list = []

    losses_dict={}

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            # input and target
            batchsize = input_tuple[0].size(0)
            # if args.gpus is not None:
            #     input_tuple = [x.cuda(args.gpus, non_blocking=True) for x in input_tuple]
            # else:
            input_tuple = [x.cuda(non_blocking=True) for x in input_tuple]
            target = input_tuple[-1]
            # target = input_tuple[-1].cuda(args.gpus, non_blocking=True)

            # model forward function
            output, mask_stack_list, _, gate_meta = \
                model(input=input_tuple[:-1], tau=tau, is_training=False, curr_step=0)

            # measure losses, accuracy and predictions
            if args.ada_reso_skip:
                upb_gflops_tensor, real_gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops, gflops_list, args)
                loss_dict = compute_losses(criterion, output, target, mask_stack_list,
                                           upb_gflops_tensor, real_gflops_tensor, epoch, model,
                                           val_loader.a_v_m, val_loader.a_n_m, base_model_gflops, args)
                upb_batch_gflops_list.append(upb_gflops_tensor)
                real_batch_gflops_list.append(real_gflops_tensor)
            else:
                if args.dataset=="epic":
                    v_acc_loss, n_acc_loss, acc_loss = compute_epic_losses(criterion, output, target,
                                                                           val_loader.a_v_m, val_loader.a_n_m)
                    loss_dict = {"loss": acc_loss, "verb_loss": v_acc_loss, "noun_loss": n_acc_loss}
                else:
                    loss_dict = {"loss": criterion(output, target[:, 0])}

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            if args.dataset == "epic":
                verb_prec1, verb_prec5, noun_prec1, noun_prec5 = verb_noun_accuracy(
                    val_loader.a_v_m, val_loader.a_n_m, output.data, target, topk=(1, 5))

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
                if args.dataset == "epic":
                    dist.all_reduce(verb_prec1)
                    dist.all_reduce(verb_prec5)
                    dist.all_reduce(noun_prec1)
                    dist.all_reduce(noun_prec5)
                    verb_prec1 /= world_size
                    verb_prec5 /= world_size
                    noun_prec1 /= world_size
                    noun_prec5 /= world_size

            all_results.append(output)
            all_targets.append(target)

            # record loss and accuracy
            if len(losses_dict) == 0:
                losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
            for loss_name in loss_dict:
                losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
            top1.update(prec1.item(), batchsize)
            top5.update(prec5.item(), batchsize)
            if args.dataset == "epic":
                verb_top1.update(verb_prec1.item(), batchsize)
                verb_top5.update(verb_prec5.item(), batchsize)
                noun_top1.update(noun_prec1.item(), batchsize)
                noun_top5.update(noun_prec5.item(), batchsize)

            if args.ada_reso_skip:
                # gather masks
                for layer_i, mask_stack in enumerate(mask_stack_list):
                    if "batenet" in args.arch:
                        mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                    else:  # TODO CGNet
                        mask_stack_list_list[layer_i].append(mask_stack.detach()) #TODO remvoed .cpu()

            if args.save_meta_gate:
                gate_meta_list.append(gate_meta.cpu())
                mask_stat=[]
                for layer_i, mask_stack in enumerate(mask_stack_list):
                    mask_stat.append(torch.sum(mask_stack.cpu(), dim=2))  # TODO: N*T*C*3 -> N*T*3
                mask_stat = torch.stack(mask_stat, dim=2)  # TODO L, N*T*3->N*T*L*3
                mask_stat_list.append(mask_stat)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.
                                format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses_dict["loss"], top1=top1, top5=top5))
                if args.dataset == "epic":
                    print_output += "V@1 {v1.val:.3f}({v1.avg:.3f}) V@5 {v5.val:.3f}({v5.avg:.3f}) " \
                                    "N@1 {n1.val:.3f}({n1.avg:.3f}) N@5 {n5.val:.3f}({n5.avg:.3f}) ". \
                        format(v1=verb_top1, v5=verb_top5, n1=noun_top1, n5=noun_top5)

                for loss_name in losses_dict:
                    if loss_name == "loss" or "mask" in loss_name:
                        continue
                    print_output += ' {header:s} {loss.val:.3f}({loss.avg:.3f})'. \
                        format(header=loss_name[0], loss=losses_dict[loss_name])
                print(print_output)
    if args.ada_reso_skip:
        if "cgnet" in args.arch:
            for layer_i in range(len(mask_stack_list_list)):
                mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
        # upb_batch_gflops, real_batch_gflops = compute_gflops_by_mask(mask_stack_list_list, base_model_gflops, gflops_list, args)
        upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
        real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue)  multi-label mAP


    if dist.is_initialized():
        mAP_tensor = torch.tensor(mAP).to(all_results[0].device)
        mmAP_tensor = torch.tensor(mmAP).to(all_results[0].device)

        world_size = dist.get_world_size()
        if args.ada_reso_skip:
            dist.all_reduce(upb_batch_gflops)
            dist.all_reduce(real_batch_gflops)
        dist.all_reduce(mAP_tensor)
        dist.all_reduce(mmAP_tensor)
        if args.ada_reso_skip:
            upb_batch_gflops /= world_size
            real_batch_gflops /= world_size
        mAP_tensor /= world_size
        mmAP_tensor /= world_size
        mAP = mAP_tensor.item()
        mmAP = mmAP_tensor.item()

    if args.rank==0:
        epic_str=""
        if args.dataset=="epic":
            epic_str = "V@1: %.3f V@5: %.3f N@1: %.3f N@5: %.3f" % (verb_top1.avg, verb_top5.avg, noun_top1.avg, noun_top5.avg)

        print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} {epic_str:s} Loss {loss.avg:.5f}'
              .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses_dict["loss"],
              epic_str=epic_str))
        if args.ada_reso_skip:
            usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
            print(usage_str)
            # if args.print_statistics:
            print_mask_statistics(mask_stack_list_list, args)
        else:
            usage_str = "Base Model"

        if args.save_meta_gate:
            #  TODO all_targets, all_preds, all_mask_stats
            all_mask_stats = torch.cat(mask_stat_list, dim=0)
            all_preds = torch.cat(gate_meta_list, dim=0)

            np.savez("%s/meta-gate-val.npy" % (exp_full_path),
                     mask_stats=all_mask_stats.numpy(), preds=all_preds.numpy(), targets=torch.cat(all_targets, 0).cpu().numpy())

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses_dict["loss"].avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
    else:
        usage_str = "Empty: non-master node"
    if args.dataset=="epic":
        return mAP, mmAP, top1.avg, top5.avg, usage_str, (verb_top1.avg, verb_top5.avg, noun_top1.avg, noun_top5.avg)
    else:
        return mAP, mmAP, top1.avg, top5.avg, usage_str, None


def save_checkpoint(state, is_best, shall_backup, exp_full_path, decorator):
    if is_best:
        torch.save(state, '%s/models/%s.pth.tar' % (exp_full_path, decorator))
    if shall_backup:
        copyfile("%s/models/ckpt.best.pth.tar"%exp_full_path,
                 "%s/models/oldbest.%s.pth.tar"%(exp_full_path, decorator))


def adjust_learning_rate(optimizer, epoch, length, iteration, lr_type, lr_steps, args):
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    elif lr_type == 'linear':
        factor = min(1.0, (epoch * length + iteration + 1)/(args.warmup_epochs * length))
        lr = args.lr * factor
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        if lr_type != 'linear':
            param_group['weight_decay'] = decay * param_group['decay_mult']


def setup_log_directory(exp_header, test_mode, args, logger):
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
        print("data_path : %s" % args.root_path)
        print("train_list: %s" % args.train_list)
        print("val_list  : %s" % args.val_list)
        print("%s: %d classes" % (args.dataset, args.num_class))

    # train_augmentation = model.module.get_augmentation(
    #     flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    train_transform_flip = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=True),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    train_transform_nofl = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=False),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    val_transform = torchvision.transforms.Compose([
                       GroupScale(int(model.module.scale_size)),
                       GroupCenterCrop(model.module.crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize(model.module.input_mean, model.module.input_std),
                   ])

    train_dataset = TSNDataSet(args.root_path, args.train_list,
                               num_segments=args.num_segments,
                               image_tmpl=prefix,
                               transform=(train_transform_flip, train_transform_nofl),
                               dense_sample=args.dense_sample,
                               dataset=args.dataset,
                               filelist_suffix=args.filelist_suffix,
                               folder_suffix=args.folder_suffix,
                               save_meta=args.save_meta,
                               always_flip=args.always_flip,
                               conditional_flip=args.conditional_flip,
                               adaptive_flip=args.adaptive_flip,
                               rank=args.rank)

    val_dataset = TSNDataSet(args.root_path, args.val_list,
                             num_segments=args.num_segments,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=(val_transform, val_transform),
                             dense_sample=args.dense_sample,
                             dataset=args.dataset,
                             filelist_suffix=args.filelist_suffix,
                             folder_suffix=args.folder_suffix,
                             save_meta=args.save_meta,
                             rank=args.rank)

    train_loader = build_dataflow(train_dataset, True, args.batch_size, args.workers, args.distributed, args.not_pin_memory)
    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.distributed, args.not_pin_memory)

    if args.dataset == "epic":
        train_loader.a_v_m = train_dataset.a_v_m
        train_loader.a_n_m = train_dataset.a_n_m
        val_loader.a_v_m = val_dataset.a_v_m
        val_loader.a_n_m = val_dataset.a_n_m
    else:
        train_loader.a_v_m = None
        train_loader.a_n_m = None
        val_loader.a_v_m = None
        val_loader.a_n_m = None

    return train_loader, val_loader

def take_care_of_pretraining(sd, args):
    old_to_new_pairs = []
    if args.downsample0_renaming:
        for k in sd:
            if "downsample0" in k:
                old_to_new_pairs.append((k, k.replace("downsample0", "downsample.0")))
            elif "downsample1" in k:
                old_to_new_pairs.append((k, k.replace("downsample1", "downsample.1")))

    if args.downsample_0_renaming or (any([args.use_segk8, args.use_tsmk8, args.use_tsmk8]) and args.ada_reso_skip):
        for k in sd:
            if "downsample.0" in k:
                old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
            elif "downsample.1" in k:
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

    for old_key, new_key in old_to_new_pairs:
        sd[new_key] = sd.pop(old_key)
    old_to_new_pairs = []

    for old_key, new_key in old_to_new_pairs:
        sd[new_key] = sd.pop(old_key)

    old_to_new_pairs = []
    if args.shift and args.ada_reso_skip:
        for k in sd:
            if "conv1.net." in k:
                old_to_new_pairs.append((k, k.replace("conv1.net.", "conv1.")))
    for old_key, new_key in old_to_new_pairs:
        sd[new_key] = sd.pop(old_key)

    del_keys = []
    if args.ignore_new_fc_weight or any([args.use_segk8, args.use_tsmk8, args.use_tsmk8]):
        del_keys += [k for k in sd if "module.new_fc" in k]
    for k in del_keys:
        del sd[k]

    del_keys = []
    if args.ignore_loading_gate_fc:
        del_keys += [k for k in sd if "gate_fc" in k]
    for k in del_keys:
        del sd[k]
    return sd

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
                if keyword[0] == "J":
                    if keyword[-1] == "J":  # TODO middle
                        if keyword[1:-1] in name:
                            param.requires_grad = False
                            if args.rank==0:
                                print(keyword, "->", name, "frozen")
                    else:  # TODO suffix
                        if name.endswith(keyword[1:]):
                            param.requires_grad = False
                            if args.rank == 0:
                                print(keyword, "->", name, "frozen")
                elif keyword[-1] == "J":  # TODO prefix
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
