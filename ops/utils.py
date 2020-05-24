import numpy as np
import torch
import torch.nn.functional as F

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_marginal_output(output, the_map, num_clusters):
    # num_clusters = len(set(the_map.values()))
    # sub_output = torch.zeros_like(output[:, :num_clusters])
    # for k in the_map:
    #     sub_output[:, the_map[k]] += output[:, k]
    sub_output = torch.matmul(output, the_map)
    return sub_output

def verb_noun_accuracy(a_v_map, a_n_map, output, target, topk):
    verb_output = get_marginal_output(output, a_v_map, 125)
    noun_output = get_marginal_output(output, a_n_map, 352)
    # print(target)
    # print(verb_output, target[:,1])
    # print(noun_output, target[:,2])
    # print("verb_out",verb_output)
    # print("noun_output",noun_output)
    verb_top1, verb_top5 = accuracy(verb_output, target[:, 1], topk)
    noun_top1, noun_top5 = accuracy(noun_output, target[:, 2], topk)
    return verb_top1, verb_top5, noun_top1, noun_top5

def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    # TODO ranking labels: (-1,-1,4,5,3,7)->(4,4,2,1,0,3)
    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1
        #TODO(DEBUG)
        # assert label_cnt == classes

    gt = torch.zeros(bs, classes + 1)  # TODO(yue) +1 for -1 in multi-label case
    for i in range(test_y.shape[1]):
        gt[torch.LongTensor(range(bs)), test_y[:, i]] = 1  # TODO(yue) see?

    return gt[:, :classes]

def cal_map(output, old_test_y):
    batch_size = output.size(0)
    num_classes = output.size(1)
    ap = torch.zeros(num_classes)
    test_y = old_test_y.clone()

    gt = get_multi_hot(test_y, num_classes, False)

    probs = F.softmax(output, dim=1)

    rg = torch.range(1, batch_size).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)

        ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
        #print(k, ap[k])
    #print()
    return ap.mean()*100, ap*100


class Recorder:
    def __init__(self, larger_is_better=True):
        self.history=[]
        self.larger_is_better=larger_is_better
        self.best_at=None
        self.best_val=None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x>y
        else:
            return x<y

    def update(self, val):
        self.history.append(val)
        if len(self.history)==1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history)-1

    def is_current_best(self):
        return self.best_at == len(self.history)-1

    def at(self, idx):
        return self.history[idx]



def count_conv2d_flops(input_data_shape, conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0
    # print("h:%d->%d"%(h_in,h_out))
    # print("c:%d->%d"%(c_in,c_out))
    flops = n * c_out * h_out * w_out * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    # print("flops",flops)
    return flops, (n, c_out, h_out, w_out)


def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod

def count_bn_flops(input_data_shape):
    flops = product(input_data_shape) * 2
    output_data_shape = input_data_shape
    return flops, output_data_shape


def count_relu_flops(input_data_shape):
    flops = product(input_data_shape) * 1
    output_data_shape = input_data_shape
    return flops, output_data_shape

# def count_max_pool(input_data_shape, maxpool):
#     n, c_in, h_in, w_in = input_data_shape
#     h_out = h_in
#     output_data_shape = n, c_out, h_out, w_out

def count_fc_flops(input_data_shape, fc):
    output_data_shape = input_data_shape[:-1] + (fc.out_features, )
    flops = product(output_data_shape) * (fc.in_features + 1)
    return flops, output_data_shape

# # for debug NaN and Inf
# def debug_check(t, name="x", soft=False):
#     # assert not ((t != t).any())
#     # assert not ((t == float("inf")).any())
#
#     if ((t != t).any()):
#         if soft:
#             return "NaN"
#         else:
#             import ipdb
#             ipdb.set_trace()
#     if ((t == float("inf")).any()):
#         if soft:
#             return "Inf"
#         else:
#             import ipdb
#             ipdb.set_trace()
#     if soft:
#         return "normal"
#
# def debug_check_list(t_list):
#     stack_list=[]
#     for k, v in t_list:
#
#         # print(k, v)
#         debug_check(v, k)
#
#         if v.requires_grad and v.grad is not None:
#             stack_list.append([k+"_grad", debug_check(v.grad, k+"_grad", soft=True)])
#     if any([x[1]!="normal" for x in stack_list]):
#         import ipdb
#         ipdb.set_trace()


# if __name__ == "__main__":
#     output=torch.tensor([[0.6,0.05,0.15,0.10,0.10,0,0,0],[0.2,0.3,0.2,0.1,0.2,0,0,0],[0.2,0.2,0.1,0.1,0.4,0,0,0]])
#     target = torch.tensor([[0,1,1],[3,0,1],[4,1,2]])
#
#     a_v_m={0:1, 1:0, 2:1, 3:0, 4:1, 5:2, 6:3}
#     a_n_m={0:1, 1:0, 2:0, 3:1, 4:2, 5:2, 6:3}
#
#     top1, top5 = accuracy(output, target[:,0], topk=(1,5))
#     verb_top1, verb_top5, noun_top1, noun_top5 = verb_noun_accuracy(a_v_m, a_n_m, output, target, topk=(1,3))
#     print(top1, top5, verb_top1, verb_top5, noun_top1, noun_top5)