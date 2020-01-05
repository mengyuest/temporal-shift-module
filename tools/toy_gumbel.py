# input [0, 1]*10
# output number of numbers between 0.4~0.7
import torch
import torch.utils.data as data
from torch import nn
import numpy as np
import torch.nn.functional as F

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

class DummyDataset(data.Dataset):
    def __init__(self, length):
        upper_threshold=0.7
        lower_threshold=0.4
        self.data_list=[]
        self.label_list=[]
        for _ in range(length):
            data=(np.random.random(10)).astype(np.float32)
            label=np.sum(np.logical_and(data>=lower_threshold, data<=upper_threshold))
            label=min(9,label)
            self.data_list.append(data)
            self.label_list.append(label)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        return torch.from_numpy(data), torch.tensor([label])

    def __len__(self):
        return len(self.data_list)

def gumbel_softmax(training, x, tau = 1.0, hard=False):
    if training:
        eps = 1e-20
        U = torch.rand(x.size())
        U = -torch.log(-torch.log(U + eps) + eps)
        r_t = x + 0.5*U
        r_t = F.softmax(r_t / tau, dim=-1)

        if not hard:
            return r_t
        else:
            shape = r_t.size()
            _, ind = r_t.max(dim=-1)
            r_hard = torch.zeros_like(r_t).view(-1, shape[-1])
            r_hard.scatter_(1, ind.view(-1, 1), 1)
            r_hard = r_hard.view(*shape)
            return (r_hard - r_t).detach() + r_t
    else:
        selected = torch.zeros_like(x)
        Q_t = torch.argmax(x, dim=1).unsqueeze(1)
        selected = selected.scatter(1, Q_t, 1)
        return selected.float()

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc0 = nn.Linear(10, 20)
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x, training=True):
        tau=1.0
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        p = self.fc2(x)
        p = torch.log(F.softmax(p, dim=1))
        r = gumbel_softmax(training, p, tau, hard=True)
        return r


def train(train_loader, model, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #indices = torch.sum(torch.matmul(output, torch.tensor([[0,1,2,3,4,5,6,7,8,9.]]).transpose(0,1)), dim=1) #output.max(dim=1)[1]
        indices = output.max(dim=1)[1].float()

        #print(indices)
        loss = torch.mean(((indices-target_var.float())*(indices-target_var.float())))

        # measure accuracy and record loss
        prec1 = torch.mean((indices.long()==target_var).float())
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, top1=top1))

def validate(val_loader, model, epoch):
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        indices = output.max(dim=1)[1]

        # measure accuracy and record loss
        prec1 = torch.mean((indices.long() == target_var).float())
        top1.update(prec1.item(), input.size(0))

        print('Epoch: [{0}][{1}/{2}]\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(val_loader), top1=top1))

    return top1.avg

batch_size=12
workers=4
learning_rate=0.1
num_epochs=100
eval_freq=1

train_set=DummyDataset(100)
val_set=DummyDataset(20)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
model = DummyModel()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

best_prec1=0

for epoch in range(num_epochs):
    train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    if (epoch + 1) % eval_freq == 0:
        prec1 = validate(val_loader, model, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Best Prec@1: %.3f\n' % (best_prec1))
