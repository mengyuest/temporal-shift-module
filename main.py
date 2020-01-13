# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import shutil
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

# TODO(yue)
import common
from os.path import join as ospj

def main():
    global args, best_prec1, num_class
    args = parser.parse_args()

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

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            mAP, mmAP, prec1 = validate(val_loader, model, criterion, epoch, tf_writer)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)

            tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)

            print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)\n' % (
                map_record.best_val, map_record.best_at,
                mmap_record.best_val, mmap_record.best_at,
                prec_record.best_val, prec_record.best_at))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': prec_record.best_val,
            }, prec_record.is_current_best(), exp_full_path)

def cal_eff(r):
    # TODO r N, T, 3 or 2

    r_loss=torch.tensor([4., 2., 1., 0.5, 0.25, 0.125, 0.0625,0.03125]).cuda()

    return torch.sum(torch.mean(r, axis=[0,1]) * r_loss[:r.shape[2]])


def reverse_onehot(a):
    return [np.where(r == 1)[0][0] for r in a]


def get_criterion_loss(criterion, output, target):
    if args.loss_type == "bce":
        multi_hot_target = get_multi_hot(target, num_class).cuda()
        return criterion(output, multi_hot_target)
    else:
        return criterion(output, target[:, 0])


def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, epoch):
    if epoch > args.eff_loss_after:
        acc_weight = args.accuracy_weight
        eff_weight = args.efficency_weight
    else:
        acc_weight = 1.0
        eff_weight = 0.0
    return acc_loss * acc_weight, eff_loss * eff_weight


def train(train_loader, model, criterion, optimizer, epoch, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
        alosses = AverageMeter()
        elosses = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, input_tuple in enumerate(train_loader):
        target = input_tuple[-1]
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input = input_tuple[0]
        if args.ada_reso_skip:
            input_var_list=[torch.autograd.Variable(input_item) for input_item in input_tuple[:-1]]
            # input_var_0 = torch.autograd.Variable(input_tuple[0])
            # input_var_1 = torch.autograd.Variable(input_tuple[1])
            #TODO(yue) + LOSS + validation part!

            output, r = model(*input_var_list)
            # output, r = model(input_var_0, input_var_1)

            acc_loss = get_criterion_loss(criterion, output, target_var)
            if args.offline_lstm_last == False and args.offline_lstm_all == False:
                eff_loss = cal_eff(r)
                acc_loss, eff_loss = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, epoch)
                alosses.update(acc_loss.item(), input.size(0))
                elosses.update(eff_loss.item(), input.size(0))
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
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO

            if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
                print_output += '\ta {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                    aloss = alosses, eloss =elosses, r=reverse_onehot(r[-1,:,:].detach().cpu().numpy())
                )

            if args.show_pred:
                print_output +="\tp {p}".format(p=output[-1,:].detach().cpu().numpy())

            print(print_output)

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)



def validate(val_loader, model, criterion, epoch, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # TODO(yue)
    all_results = []
    all_targets = []

    if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
        alosses = AverageMeter()
        elosses = AverageMeter()
        r_list=[]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            target = input_tuple[-1].cuda()
            input = input_tuple[0]

            # compute output
            if args.ada_reso_skip:
                #output, r = model(input_tuple[0], input_tuple[1])
                output, r = model(*input_tuple[:-1])
                acc_loss = get_criterion_loss(criterion, output, target)
                if args.offline_lstm_last == False and args.offline_lstm_all == False:
                    eff_loss = cal_eff(r)
                    acc_loss, eff_loss = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, epoch)
                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
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
            if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
                r_list.append(r.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
                    output += '\ta {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                        aloss=alosses, eloss=elosses, r=reverse_onehot(r[-1, :, :].cpu().numpy())
                    )

                print(output)

    # TODO(yue)
    mAP,_ = cal_map(torch.cat(all_results,0).cpu(), torch.cat(all_targets,0)[:,0:1].cpu()) # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())    # TODO(yue)  multi-label mAP

    output = ('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))
    print(output)

    if args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False:
        rs=np.concatenate(r_list, axis=0)

        tmp_cnt = [np.sum(rs[:, :, iii]==1) for iii in range(rs.shape[2])]
        tmp_total_cnt = sum(tmp_cnt)

        for action_i in range(rs.shape[2]):
            action_str = "scale_%d"%(action_i) if action_i<len(args.reso_list) else "skip__%d"%(action_i-len(args.reso_list)+1)
            print("action %s: %5d (%.4f)"%(action_str,tmp_cnt[action_i],tmp_cnt[action_i]/tmp_total_cnt))

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return mAP, mmAP, top1.avg


def save_checkpoint(state, is_best, exp_full_path):
    filename = '%s/models/ckpt.pth.tar' % (exp_full_path)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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
    exp_full_path = ospj(common.EXPS_PATH, exp_full_name)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path,"models"))
    logger.create_log(exp_full_path)
    return exp_full_path

if __name__ == '__main__':
    main()
