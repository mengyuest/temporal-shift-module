# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.models_ada import TSN_Ada
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, get_multi_hot
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
from ops.my_logger import Logger


best_prec1 = 0
num_class = -1
use_ada_framework = False
NUM_LOSSES=10

# TODO(yue)
import common
from os.path import join as ospj

def load_to_sd(model_dict, model_path, module_name, fc_name, resolution, apple_to_apple=False):
    if ".pth" in model_path:
        print("done loading\t%s\t(res:%3d) from\t%s" % ("%-25s"%module_name, resolution, model_path))
        if os.path.exists(common.PRETRAIN_PATH+ "/" +model_path):
            sd = torch.load(common.PRETRAIN_PATH+ "/" +model_path)['state_dict']
        elif os.path.exists(common.EXPS_PATH+ "/" +model_path):
            print("Didn't find in %s, check %s"%(common.PRETRAIN_PATH, common.EXPS_PATH))
            sd = torch.load(common.EXPS_PATH + "/" + model_path)['state_dict']
            print("Success~")
        else:
            exit("Cannot find model, exit")
        if apple_to_apple:
            return sd

        replace_dict = []
        nowhere_ks = []
        notfind_ks = []

        for k, v in sd.items():  # TODO(yue) base_model->base_model_list.i
            new_k = k.replace("base_model", module_name)
            new_k = new_k.replace("new_fc", fc_name)
            if new_k in model_dict:
                replace_dict.append((k, new_k))
            else:
                nowhere_ks.append(k)
        for new_k, v in model_dict.items():
            if module_name in new_k:
                k = new_k.replace(module_name, "base_model")
                if k not in sd:
                    notfind_ks.append(k)
            if fc_name in new_k:
                k = new_k.replace(fc_name, "new_fc")
                if k not in sd:
                    notfind_ks.append(k)
        if len(nowhere_ks) != 0:
            print("Vars not in ada network, but are in pretrained weights\n" + ("\n%sNEW  "%module_name).join(nowhere_ks))
        if len(notfind_ks) != 0:
            print("Vars not in pretrained weights, but are needed in ada network\n" + ("\n%sLACK "%module_name).join(notfind_ks))
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        return {k: v for k, v in sd.items() if k in model_dict}
    else:
        print("skip loading\t%s\t(res:%3d) from\t%s"%("%-25s"%module_name, resolution, model_path))
        return {}

def main():
    t_start = time.time()
    global args, best_prec1, num_class, use_ada_framework
    args = parser.parse_args()

    set_random_seed(args.random_seed)
    use_ada_framework = args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False

    logger=Logger()
    sys.stdout = logger

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'

    if args.ada_reso_skip:
        MODEL_NAME = TSN_Ada
    else:
        MODEL_NAME = TSN
    model = MODEL_NAME(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                rescale_to=args.rescale_to,
                rescale_pattern = args.rescale_pattern,
                args = args)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # TODO(yue) freeze some params in the policy+lstm layers
    if args.freeze_policy:
        for name, param in model.module.named_parameters():
            if "lite_fc" in name or "lite_backbone" in name or "rnn" in name or "linear" in name:
                param.requires_grad = False

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    # TODO(yue) ada_model loading process
    if args.ada_reso_skip:
        if args.base_pretrained_from != "":
            print("Adaptively load from pretrained whole")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, args.base_pretrained_from, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)

        elif len(args.model_paths)!=0:
            print("Adaptively load from model_path_list")
            model_dict=model.state_dict()
            # TODO(yue) policy net
            sd = load_to_sd(model_dict, args.policy_path, "lite_backbone", "lite_fc", args.reso_list[args.policy_input_offset])
            model_dict.update(sd)
            # TODO(yue) backbones
            for i, tmp_path in enumerate(args.model_paths):
                sd = load_to_sd(model_dict, tmp_path, "base_model_list.%d"%i, "new_fc_list.%d"%i, args.reso_list[i])
                model_dict.update(sd)

            model.load_state_dict(model_dict)
        print()

    if args.ada_reso_skip == False and args.base_pretrained_from != "":
        print("Baseline: load from pretrained model")
        model_dict = model.state_dict()
        sd = load_to_sd(model_dict, args.base_pretrained_from, "base_model", "new_fc", 224)
        model_dict.update(sd)
        model.load_state_dict(model_dict)
        print()

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   partial_fcvid_eval=False,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, args=args),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, args=args),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_type == "bce":
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    exp_full_path = setup_log_directory(logger, args.exp_header)

    with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=exp_full_path)

    # TODO(yue)
    map_record = Recorder()
    mmap_record = Recorder()
    prec_record = Recorder()
    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(args.random_seed+epoch)
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, tf_writer)
        else:
            train_usage_str = "No training usage stats (Eval Mode)"

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed)
            mAP, mmAP, prec1, val_usage_str = validate(val_loader, model, criterion, epoch, tf_writer)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)

            if mmap_record.is_current_best():
                best_train_usage_str = train_usage_str
                best_val_usage_str = val_usage_str

            print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)\n' % (
                map_record.best_val, map_record.best_at,
                mmap_record.best_val, mmap_record.best_at,
                prec_record.best_val, prec_record.best_at))

            if args.skip_training:
                break

            tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': prec_record.best_val,
            }, map_record.is_current_best(), exp_full_path)

    if use_ada_framework:
        print("Best train usage:")
        print(best_train_usage_str)
        print()
        print("Best val usage:")
        print(best_val_usage_str)

    print("Finished in %.4f seconds"%(time.time() - t_start))

def set_random_seed(the_seed):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)


def get_gflops_t_tt_vector():
    x2_coeff = {"resnet18": 36144, "resnet34": 73008, "resnet50": 76848, "resnet101": 150832,
                "efficientnet-b0":7772.6403, "efficientnet-b1":13950.8929, "efficientnet-b2":19929.8469,
                "efficientnet-b3":35873.7245, "efficientnet-b4":83705.3571, "efficientnet-b5":197305.4847,
                "shufflenet3d_0.5": 42, "shufflenet3d_1.0": 125, "shufflenet3d_1.5": 235, "shufflenet3d_2.0": 393,
                "res3d18": 1143.89, "res3d34": 2022.11, "res3d50": 1500.74, "res3d101": 2266.04,
                "mobilenet3dv2": 549.19
                }
    bias = {"resnet18":512000, "resnet34": 512000, "resnet50": 512000, "resnet101": 512000,
                "efficientnet-b0":0, "efficientnet-b1":0, "efficientnet-b2":0,
                "efficientnet-b3":0, "efficientnet-b4":0, "efficientnet-b5":0,
                "shufflenet3d_0.5": 0, "shufflenet3d_1.0": 0, "shufflenet3d_1.5": 0, "shufflenet3d_2.0": 0,
                "res3d18":0, "res3d34":0, "res3d50":0, "res3d101":0,
                "mobilenet3dv2":0}
    gflops_vec = []
    t_vec = []
    tt_vec = []

    for i, backbone in enumerate(args.backbone_list):
        if all([arch_name not in backbone for arch_name in ["resnet","efficientnet","shufflenet3d","mobilenet", "res3d"]]):
            exit("We can only handle resnet/mobilenet/efficientnet/shufflenet3d as backbone, when computing FLOPS")
        if args.cnn3d:
            the_flops = x2_coeff[backbone] * args.reso_list[i] * args.reso_list[i] * args.seg_len / 112. / 112. / 16. / 1000.
        else:
            the_flops = (x2_coeff[backbone] * args.reso_list[i] * args.reso_list[i] + bias[backbone]) / 1000000000.0
        gflops_vec.append(the_flops)
        t_vec.append(1.)
        tt_vec.append(1.)

    if args.policy_also_backbone:
        gflops_vec.append(0)
        t_vec.append(1.)
        tt_vec.append(1.)

    for i,_ in enumerate(args.skip_list):
        t_vec.append(1. if args.skip_list[i]==1 else 1./args.skip_list[i])
        tt_vec.append(0)
        gflops_vec.append(0)

    return gflops_vec, t_vec, tt_vec


def cal_eff(r):
    each_losses=[]
    # TODO r N * T * (#reso+#policy+#skips)
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    t_vec = torch.tensor(t_vec).cuda()
    if args.use_gflops_loss:
        r_loss = torch.tensor(gflops_vec).cuda()
    else:
        r_loss = torch.tensor([4., 2., 1., 0.5, 0.25, 0.125, 0.0625,0.03125]).cuda()[:r.shape[2]]

    loss = torch.sum(torch.mean(r, dim=[0,1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())

    #TODO(yue) uniform loss
    if args.uniform_loss_weight > 1e-5:
        usage_bias = torch.mean(r,dim=[0,1])-torch.mean(r,dim=[0,1,2])
        uniform_loss = torch.norm(usage_bias, p=2) * args.uniform_loss_weight
        loss = loss + uniform_loss
        each_losses.append(uniform_loss.detach().cpu().item())

    #TODO(yue) high-reso punish loss
    if args.head_loss_weight > 1e-5:
        head_usage = torch.mean(r[:, :, 0])
        usage_threshold=0.2
        head_loss = (head_usage - usage_threshold) * (head_usage - usage_threshold) * args.head_loss_weight
        loss = loss + head_loss
        each_losses.append(head_loss.detach().cpu().item())

    #TODO(yue) frames loss
    if args.frames_loss_weight > 1e-5:
        num_frames = torch.mean(torch.mean(r, dim=[0,1]) * t_vec)
        frames_loss = num_frames * num_frames * args.frames_loss_weight
        loss = loss + frames_loss
        each_losses.append(frames_loss.detach().cpu().item())

    return loss, each_losses


def reverse_onehot(a):
    try:
        return np.array([np.where(r > 0.5)[0][0] for r in a])
    except Exception as e:
        print("error stack:",e)
        print(a)
        for i, r in enumerate(a):
            print(i, r)
        return None


def get_criterion_loss(criterion, output, target):
    if args.loss_type == "bce":
        multi_hot_target = get_multi_hot(target, num_class).cuda()
        return criterion(output, multi_hot_target)
    else:
        return criterion(output, target[:, 0])


def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch):
    if epoch > args.eff_loss_after:
        acc_weight = args.accuracy_weight
        eff_weight = args.efficency_weight
    else:
        acc_weight = 1.0
        eff_weight = 0.0
    return acc_loss * acc_weight, eff_loss * eff_weight, [x * eff_weight for x in each_losses]

def compute_every_losses(r, acc_loss, epoch):
    eff_loss, each_losses = cal_eff(r)
    acc_loss, eff_loss, each_losses = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch)
    return acc_loss, eff_loss, each_losses


def elastic_list_print(l, limit=8):
    limit = min(limit, len(l))
    l_output = "[%s," % (",".join([str(x) for x in l[:limit//2]]))
    if l.shape[0] > limit:
        l_output += "..."
    l_output += "%s]" % (",".join([str(x) for x in l[-limit//2:]]))
    return l_output

def compute_exp_decay_tau(epoch):
    return args.init_tau * np.exp(args.exp_decay_factor * epoch)

def get_policy_usage_str(r_list, reso_dim):
    printed_str=""
    rs = np.concatenate(r_list, axis=0)

    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()

    tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])]
    tmp_total_cnt = sum(tmp_cnt)

    gflops = 0
    avg_frame_ratio = 0
    num_preds = 0

    for action_i in range(rs.shape[2]):
        if args.policy_also_backbone and action_i == reso_dim - 1:
            action_str = "m0(%s %dx%d)" % (args.policy_backbone, args.reso_list[args.policy_input_offset], args.reso_list[args.policy_input_offset])
        elif action_i < reso_dim:
            action_str = "r%d(%7s %dx%d)" % (action_i, args.backbone_list[action_i], args.reso_list[action_i], args.reso_list[action_i])
        else:
            action_str = "s%d (skip %d frames)" % (action_i - reso_dim, args.skip_list[action_i - reso_dim])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)\n" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)

        gflops += usage_ratio * gflops_vec[action_i]
        avg_frame_ratio += usage_ratio * t_vec[action_i]
        num_preds += usage_ratio * tt_vec[action_i]

    if args.cnn3d:
        if args.policy_backbone!="shufflenet3d_0.5":
            exit("if cnn3d, only using shufflenet3d_0.5 as policy net")
        # TODO (policy backbone-shufflenet3d)
        gflops += 0.003 * avg_frame_ratio / 112 / 112 * args.reso_list[args.policy_input_offset] * args.reso_list[args.policy_input_offset]
        gflops += 0.0004 * avg_frame_ratio  # TODO (lstm)
        time_steps = args.num_segments // args.seg_len
    else:
        gflops += 0.08 * avg_frame_ratio #TODO (policy backbone)
        gflops += 0.01 * avg_frame_ratio #TODO (lstm)
        time_steps = args.num_segments

    printed_str += "GFLOPS: %.6f  AVG_FRAMES: %.3f  NUM_PREDS: %.3f"%(gflops, avg_frame_ratio*args.num_segments, num_preds * time_steps)

    return printed_str

def extra_each_loss_str(each_terms):
    loss_str_list = ["gf"]
    s = ""
    if args.uniform_loss_weight > 1e-5:
        loss_str_list.append("u")
    if args.head_loss_weight > 1e-5:
        loss_str_list.append("h")
    if args.frames_loss_weight > 1e-5:
        loss_str_list.append("f")
    for i in range(len(loss_str_list)):
        s += " %s:%.4f" % (loss_str_list[i], each_terms[i].val)
    return s

def get_current_temperature(num_epoch):
    if args.exp_decay:
        tau = compute_exp_decay_tau(num_epoch)
    else:
        tau = args.init_tau
    return tau

def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]

def train(train_loader, model, criterion, optimizer, epoch, tf_writer):
    batch_time, data_time, losses, top1, top5 = get_average_meters(5)
    tau=0
    if use_ada_framework:
        tau = get_current_temperature(epoch)
        alosses, elosses = get_average_meters(2)
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []


    model.module.partialBN(not args.no_partialbn)

    # switch to train mode
    model.train()

    end = time.time()
    print("lr:%.4f\ttau:%.4f"%(optimizer.param_groups[-1]['lr'] * 0.1, tau if use_ada_framework else 0))
    for i, input_tuple in enumerate(train_loader):

        data_time.update(time.time() - end)  # TODO(yue) measure data loading time

        target = input_tuple[-1].cuda()
        target_var = torch.autograd.Variable(target)
        input = input_tuple[0]
        if args.ada_reso_skip:
            input_var_list=[torch.autograd.Variable(input_item) for input_item in input_tuple[:-1]]

            output, r = model(input=input_var_list, tau=tau)

            acc_loss = get_criterion_loss(criterion, output, target_var)
            if use_ada_framework:
                acc_loss, eff_loss, each_losses = compute_every_losses(r, acc_loss, epoch)
                alosses.update(acc_loss.item(), input.size(0))
                elosses.update(eff_loss.item(), input.size(0))
                for l_i, each_loss in enumerate(each_losses):
                    each_terms[l_i].update(each_loss, input.size(0))
                loss = acc_loss + eff_loss
            else:
                loss = acc_loss
        else:
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            loss = get_criterion_loss(criterion, output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target[:,0], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if use_ada_framework:
            r_list.append(r.detach().cpu().numpy())

        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))  # TODO

            if use_ada_framework:
                print_output += ' a {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                    aloss = alosses, eloss =elosses, r=elastic_list_print(reverse_onehot(r[-1,:,:].detach().cpu().numpy()))
                )
                print_output += extra_each_loss_str(each_terms)

            if args.show_pred:
                print_output += elastic_list_print(output[-1,:].detach().cpu().numpy())
            print(print_output)

    if use_ada_framework:
        usage_str = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

    return usage_str if use_ada_framework else None



def validate(val_loader, model, criterion, epoch, tf_writer=None):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau=0
    # TODO(yue)
    all_results = []
    all_targets = []

    if use_ada_framework:
        tau = get_current_temperature(epoch)
        alosses, elosses = get_average_meters(2)
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            target = input_tuple[-1].cuda()
            input = input_tuple[0]

            # compute output
            if args.ada_reso_skip:
                output, r = model(input=input_tuple[:-1], tau=tau)
                acc_loss = get_criterion_loss(criterion, output, target)
                if use_ada_framework:
                    acc_loss, eff_loss, each_losses = compute_every_losses(r, acc_loss, epoch)
                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
                    for l_i, each_loss in enumerate(each_losses):
                        each_terms[l_i].update(each_loss, input.size(0))
                    loss = acc_loss + eff_loss
                else:
                    loss = acc_loss
            else:
                output = model(input)
                loss = get_criterion_loss(criterion, output, target)

            # TODO(yue)
            all_results.append(output)
            all_targets.append(target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:,0], topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if use_ada_framework:
                r_list.append(r.cpu().numpy())

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                if use_ada_framework:
                    print_output += ' a {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(reverse_onehot(r[-1, :, :].cpu().numpy()))
                    )

                    print_output += extra_each_loss_str(each_terms)

                print(print_output)

    # TODO(yue)
    mAP,_ = cal_map(torch.cat(all_results,0).cpu(), torch.cat(all_targets,0)[:,0:1].cpu()) # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())    # TODO(yue)  multi-label mAP

    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))

    if use_ada_framework:
        usage_str = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return mAP, mmAP, top1.avg, usage_str if use_ada_framework else None

def save_checkpoint(state, is_best, exp_full_path):
    filename = '%s/models/ckpt%04d.pth.tar' % (exp_full_path, state["epoch"])
    # if (state["epoch"]-1) % args.save_freq == 0 or state["epoch"] == 0:
    #     torch.save(state, filename)
    if is_best:
        torch.save(state, '%s/models/ckpt.best.pth.tar' % (exp_full_path))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
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


def setup_log_directory(logger, exp_header):
    exp_full_name = "g%s_%s"%(logger._timestr, exp_header)
    if args.skip_training:
        exp_full_name+="_test"
    exp_full_path = ospj(common.EXPS_PATH, exp_full_name)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path,"models"))
    logger.create_log(exp_full_path)
    return exp_full_path



if __name__ == '__main__':
    main()

