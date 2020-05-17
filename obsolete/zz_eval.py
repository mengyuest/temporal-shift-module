import warnings
warnings.filterwarnings("ignore")
import os
import time
from os.path import join as ospj
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from obsolete.ops.zz_data_loader import ARNetDataSet
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map
from tools.net_flops_table import get_gflops_params, feat_dim_dict
from obsolete.ops.zz_arnet import ARNet
import common


best_prec1 = 0
num_class = -1
NUM_LOSSES=10
gflops_table = {}

def load_to_sd(model_path):
    if os.path.exists(common.EXPS_PATH+ "/" +model_path):
        return torch.load(common.EXPS_PATH + "/" + model_path)['state_dict']
    exit("Cannot find model. Exit...")


def set_random_seed(the_seed):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)


def init_gflops_table():
    global gflops_table
    gflops_table = {}
    for i, backbone in enumerate(args.backbone_list):
        gflops_table[backbone+str(args.reso_list[i])] = get_gflops_params(backbone, args.reso_list[i], num_class, -1)[0]
    gflops_table["policy"] = get_gflops_params(args.policy_backbone, args.reso_list[args.policy_input_offset], num_class, -1)[0]
    gflops_table["lstm"] = 2 * (feat_dim_dict[args.policy_backbone] ** 2) /1000000000
    print("gflops_table: ")
    for k in gflops_table:
        print("%-20s: %.4f GFLOPS"%(k,gflops_table[k]))


def get_gflops_t_tt_vector():
    gflops_vec = []
    t_vec = []
    tt_vec = []

    for i, backbone in enumerate(args.backbone_list):
        if all([arch_name not in backbone for arch_name in ["resnet","mobilenet", "efficientnet"]]):
            exit("We can only handle resnet/mobilenet/efficientnet as backbone, when computing FLOPS")

        the_flops = gflops_table[backbone+str(args.reso_list[i])]
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
    r_loss = torch.tensor(gflops_vec).cuda()
    loss = torch.sum(torch.mean(r, dim=[0,1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())

    #TODO(yue) uniform loss
    if args.uniform_loss_weight > 1e-5:
        if_policy_backbone = 1 if args.policy_also_backbone else 0
        num_pred = len(args.backbone_list)
        policy_dim = num_pred + if_policy_backbone + len(args.skip_list)

        reso_skip_vec=torch.zeros(policy_dim).cuda()

        for b_i in range(num_pred):
            reso_skip_vec[b_i] += torch.sum(r[:, :, b_i:b_i+1])

        #TODO mobilenet + skips
        for b_i in range(num_pred, reso_skip_vec.shape[0]):
            reso_skip_vec[b_i] = torch.sum(r[:, :, b_i])

        reso_skip_vec = reso_skip_vec / torch.sum(reso_skip_vec)
        usage_bias = reso_skip_vec - torch.mean(reso_skip_vec)
        uniform_loss = torch.norm(usage_bias, p=2) * args.uniform_loss_weight
        loss = loss + uniform_loss
        each_losses.append(uniform_loss.detach().cpu().item())

    return loss, each_losses


def reverse_onehot(a):
    return np.array([np.where(r > 0.5)[0][0] for r in a])


def get_criterion_loss(criterion, output, target):
    return criterion(output, target[:, 0])


def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses):
    return acc_loss * args.accuracy_weight, eff_loss * args.efficency_weight, \
           [x * args.efficency_weight for x in each_losses]


def compute_every_losses(r, acc_loss):
    eff_loss, each_losses = cal_eff(r)
    acc_loss, eff_loss, each_losses = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses)
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

    if args.all_policy:
        tmp_total_cnt = tmp_cnt[0]
    else:
        tmp_total_cnt = sum(tmp_cnt)

    gflops = 0
    avg_frame_ratio = 0
    avg_pred_ratio = 0

    for action_i in range(rs.shape[2]):
        if args.policy_also_backbone and action_i == reso_dim - 1:
            action_str = "m0(%s %d)" % (args.policy_backbone, args.reso_list[args.policy_input_offset])
        elif action_i < reso_dim:
            action_str = "r%d(%7s %d)" % (action_i, args.backbone_list[action_i], args.reso_list[action_i])
        else:
            action_str = "s%d (skip %d frames)" % (action_i - reso_dim, args.skip_list[action_i - reso_dim])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)\n" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)

        gflops += usage_ratio * gflops_vec[action_i]
        avg_frame_ratio += usage_ratio * t_vec[action_i]
        avg_pred_ratio += usage_ratio * tt_vec[action_i]

    gflops += (gflops_table["policy"] + gflops_table["lstm"]) * avg_frame_ratio
    printed_str += "GFLOPS: %.6f "%(gflops)
    return printed_str, gflops

def extra_each_loss_str(each_terms):
    loss_str_list = ["gf"]
    s = ""
    if args.uniform_loss_weight > 1e-5:
        loss_str_list.append("u")
    for i in range(len(loss_str_list)):
        s += " %s:(%.4f)" % (loss_str_list[i], each_terms[i].avg)
    return s

def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]

def validate(val_loader, model, criterion):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau=0
    all_results = []
    all_targets = []

    if args.ada_reso_skip:
        tau = args.init_tau
        alosses, elosses = get_average_meters(2)
        all_bb_results = [[] for _ in range(len(args.backbone_list))]
        if args.policy_also_backbone:
            all_bb_results.append([])
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            target = input_tuple[-1].cuda()
            input = input_tuple[0]
            # compute output
            if args.ada_reso_skip:
                output, r, feat_outs, base_outs = model(input=input_tuple[:-1], tau=tau)
                acc_loss = get_criterion_loss(criterion, output, target)
                if args.ada_reso_skip:
                    acc_loss, eff_loss, each_losses = compute_every_losses(r, acc_loss)
                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
                    for l_i, each_loss in enumerate(each_losses):
                        each_terms[l_i].update(each_loss, input.size(0))
                    loss = acc_loss + eff_loss
                else:
                    loss = acc_loss
            else:
                output = model(input=[input])
                loss = get_criterion_loss(criterion, output, target)

            # TODO(yue)
            all_results.append(output)
            all_targets.append(target)

            if args.ada_reso_skip:
                for bb_i in range(len(all_bb_results)):
                    all_bb_results[bb_i].append(base_outs[:, bb_i])

            prec1, prec5 = accuracy(output.data, target[:,0], topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.ada_reso_skip:
                r_list.append(r.cpu().numpy())

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                if args.ada_reso_skip:
                    print_output += ' a {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(reverse_onehot(r[-1, :, :].cpu().numpy()))
                    )
                    print_output += extra_each_loss_str(each_terms)
                print(print_output)

    mAP,_ = cal_map(torch.cat(all_results,0).cpu(), torch.cat(all_targets,0)[:,0:1].cpu()) # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())    # TODO(yue)  multi-label mAP
    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))

    if args.ada_reso_skip:
        bbmmaps = []
        bbprec1s = []
        all_targets_cpu=torch.cat(all_targets, 0).cpu()
        for bb_i in range(len(all_bb_results)):
            bb_results_cpu=torch.mean(torch.cat(all_bb_results[bb_i], 0), dim=1).cpu()
            bb_i_mmAP, _ = cal_map(bb_results_cpu, all_targets_cpu)  # TODO(yue)  multi-label mAP
            bbmmaps.append(bb_i_mmAP)

            bbprec1, = accuracy(bb_results_cpu, all_targets_cpu[:, 0], topk=(1,))
            bbprec1s.append(bbprec1)

        print("bbmmAP: "+" ".join(["{0:.3f}".format(bb_i_mmAP) for bb_i_mmAP in bbmmaps]))
        print("bb_Acc: "+" ".join(["{0:.3f}".format(bbprec1) for bbprec1 in bbprec1s]))
        usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)


def main():
    t_start = time.time()
    global args, best_prec1, num_class

    set_random_seed(args.random_seed)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset)
    if args.ada_reso_skip:
        init_gflops_table()

    # model
    model = ARNet(num_class, args.num_segments, args.arch, args.pretrain, args)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # load weights
    model_dict = model.state_dict()
    sd = load_to_sd(ospj(args.test_from, "models", "ckpt.best.pth.tar"))
    model_dict.update(sd)
    model.load_state_dict(model_dict)
    cudnn.benchmark = True

    # transform
    val_transform = torchvision.transforms.Compose([
        GroupScale(int(model.module.scale_size)),
        GroupCenterCrop(model.module.crop_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    # dataloader
    val_loader = torch.utils.data.DataLoader(
        ARNetDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   image_tmpl=prefix, transform=val_transform, args=args),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # validatation
    set_random_seed(args.random_seed)
    validate(val_loader, model, torch.nn.CrossEntropyLoss().cuda())
    print("Finished in %.4f seconds\n" % (time.time() - t_start))


if __name__ == '__main__':
    args = parser.parse_args()
    print("======== TEST MODE ========")
    main()