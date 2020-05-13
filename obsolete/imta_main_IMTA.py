#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from obsolete.imta_dataloader import get_dataloaders
from obsolete.imta_args import arg_parser
from obsolete.imta_adaptive_inference import dynamic_evaluate
from obsolete import imta_models

from ops.utils import cal_map
import common
from os.path import join as ospj
import warnings  # TODO(yue)
warnings.filterwarnings("ignore")  # TODO(yue)

args = arg_parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
elif args.data == 'dummy':  # TODO (yue)
    args.num_classes = 10
elif args.data == 'actnet':  # TODO (yue)
    args.num_classes = 200
else:
    args.num_classes = 1000

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from obsolete.imta_utils import *

torch.manual_seed(args.seed)

# TODO(Yue) Overrided the logger
class Logger(object):
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self.log = open(log_path, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():

    global args
    best_acc1, best_epoch = 0., 0

    if args.data.startswith('cifar'): 
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    args.save = common.EXPS_PATH + "/imta/" + args.save  # TODO(Yue)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    sys.stdout = Logger(os.path.join(args.save, 'log.txt'))  # TODO(Yue)

    model = getattr(imta_models, args.arch)(args)

    if not os.path.exists(os.path.join(args.save, 'args.pth')): 
        torch.save(args, os.path.join(args.save, 'args.pth'))

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args.gpu.split(",")]).cuda()  # TODO(yue)

    # define loss function (criterion) and pptimizer
    for param in model.module.net.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD([
        {'params': model.module.classifier.parameters()},
        {'params': model.module.isc_modules.parameters()}
                                ],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    kd_loss = KDLoss(args)
    
    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)
    print("*************************************")
    print(args.use_valid, len(train_loader), len(val_loader), len(test_loader))
    print("*************************************")

    if args.evalmode is not None:
        m = torch.load(args.evaluate_from)
        model.load_state_dict(m['state_dict'])

        if args.evalmode == 'anytime':
            validate(test_loader, model, kd_loss)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    if args.imagenet_pretrained:   # TODO(yue)
        sd = torch.load(ospj(common.PYTORCH_CKPT_DIR, "msdnet-step4-block5.pth"))["state_dict"]
        state_dict = model.state_dict()
        new_sd = {k.replace("module.","module.net."): sd[k] for k in sd if "classifier" not in k}
        state_dict.update(new_sd)
        model.load_state_dict(state_dict)

    # set up logging
    global log_print

    def log_print(*args):
        print(*args)
    log_print('args:')
    log_print(args)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_acc1'
              '\tval_acc1\ttrain_acc5\tval_acc5']

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train_loss, train_acc1, train_acc5, lr = train(train_loader, model, kd_loss, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(test_loader, model, kd_loss)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_acc1, val_acc1, train_acc5, val_acc5))

        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
            best_epoch = epoch
            print('Best var_acc1 {}'.format(best_acc1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    print('Best val_acc1: {:.4f} at epoch {}'.format(best_acc1, best_epoch))

def train(train_loader, model, kd_loss, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        if args.data == 'dummy':  # TODO (yue)
            input = torch.zeros(args.batch_size, 3, 224, 224)
            target = torch.randint(0, args.num_classes, (args.batch_size,))
        elif args.data == 'actnet':  # TODO (yue)
            # print(input.shape, target.shape)
            # exit()
            _b, _tc, _h, _w = input.shape
            input = input.view(_b, _tc//3, 3, _h, _w).view(_b * _tc //3, 3, _h, _w)
            # input = input[0].view(-1,args.num_segment, 3,) # B*(TC)*H*W
            target = target[:,0]

        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        # measure data loading time
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, soft_target = model(input_var)
        if not isinstance(output, list):
            output = [output]

        if args.data == 'actnet':  # TODO (yue)
            output = [x.view(_b, _tc//3, args.num_classes).mean(dim=1) for x in output]
            soft_target = soft_target.view(_b, _tc//3, args.num_classes).mean(dim=1)

        loss = kd_loss.loss_fn_kd(output, target_var, soft_target)
        losses.update(loss.item(), input.size(0))
        for j in range(len(output)):
            acc1, acc5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(acc1.item(), input.size(0))
            top5[j].update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}](lr={3:.5f})\t'
                  'Time {batch_time.avg:.3f}({batch_time.avg:.3f})\t'
                  'Data {data_time.avg:.3f}({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.4f}({top1.avg:.4f})\t'
                  'Acc@5 {top5.val:.4f}({top5.avg:.4f})'.format(
                    epoch, i + 1, len(train_loader), lr,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, kd_loss):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # TODO (yue)
    pred_list_list=[]
    target_list=[]
    mtarget_list=[]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.data == 'dummy':
                input = torch.zeros(args.batch_size, 3, 224, 224)
                target = torch.randint(0, args.num_classes, (args.batch_size,))
            elif args.data == 'actnet':  # TODO (yue)
                _b, _tc, _h, _w = input.shape
                input = input.view(_b, _tc // 3, 3, _h, _w).view(_b * _tc // 3, 3, _h, _w)
                mtarget = target.cuda(non_blocking=True)
                target = target[:,0]
            if args.data != 'actnet':
                mtarget = target.unsqueeze(-1).cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            # compute output
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            if args.data == 'actnet':  # TODO (yue)
                output = [x.view(_b, _tc // 3, args.num_classes).mean(dim=1) for x in output]
            if len(pred_list_list)==0:  # TODO (yue)
                pred_list_list = [[] for _ in range(len(output))]

            loss = kd_loss.loss_fn_kd(output, target_var, output[-1])

            for j in range(len(output)):
                pred_list_list[j].append(output[j])  # TODO (yue)
            target_list.append(target)  # TODO (yue)
            mtarget_list.append(mtarget)  # TODO (yue)

            # measure error and record loss
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                acc1, acc5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(acc1.item(), input.size(0))
                top5[j].update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}({batch_time.avg:.3f})\t'
                      'Data {data_time.avg:.3f}({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.4f}({top1.avg:.4f})\t'
                      'Acc@5 {top5.val:.4f}({top5.avg:.4f})'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
                # break
    for j in range(args.nBlocks):
        # TODO(yue)
        mAP, _ = cal_map(torch.cat(pred_list_list[j], 0).cpu(),
                         torch.cat(mtarget_list, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
        mmAP, _ = cal_map(torch.cat(pred_list_list[j], 0).cpu(),
                          torch.cat(mtarget_list, 0).cpu())  # TODO(yue)  multi-label mAP
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  mAP {mAP:.3f}  mmAP {mmAP:.3f}'.
              format(top1=top1[j], top5=top5[j], mAP=mAP, mmAP=mmAP))  #TODO(yue)
        """
        print('Exit {}\t'
              'Err@1 {:.4f}\t'
              'Err@5 {:.4f}'.format(
              j, top1[j].avg, top5[j].avg))
        """
    # print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

def save_checkpoint(state, args, is_best, filename, result):
    # print(args)  #TODO(yue)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    if is_best:  #TODO(yue)
        os.makedirs(args.save, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, best_filename)
        with open(result_filename, 'w') as f:
            print('\n'.join(result), file=f)
        with open(latest_filename, 'w') as fout:
            fout.write(best_filename)
        print("=> saved checkpoint '{}'".format(best_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        # res.append(100.0 - correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    elif method == 'piecewise':  # TODO(yue)
        lr_steps = [0]+args.lr_steps + [100086]
        t=0
        while epoch >= lr_steps[t+1]:
            t += 1
        lr=args.lrs[t]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()

