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
