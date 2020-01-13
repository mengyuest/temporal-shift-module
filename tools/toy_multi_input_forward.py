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
    def __init__(self, the_nums, length):
        self.data_list=[]
        self.label_list=[]
        for _ in range(length):
            data=(np.random.random(the_nums)).astype(np.float32)
            label=np.sum(data[::2])+np.sum(data[1::2])*2
            self.data_list.append(data)
            self.label_list.append(label)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        return torch.from_numpy(data), torch.tensor([label])

    def __len__(self):
        return len(self.data_list)

class DummyModel(nn.Module):
    def __init__(self, the_nums):
        super(DummyModel, self).__init__()
        self.fc0 = nn.Linear(the_nums, 10)
        self.fc1 = nn.Linear(10, 1)

    def forward(self, *argv, **kwargs):
        result_list=[]
        for x in argv:
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))
            result_list.append(x)
        return tuple(result_list)


def train(train_loader, model, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    #print("flags1")
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #print("flags2")
        # compute output
        output1, output2, output3 = model(input_var, input_var, input_var)
        #indices = torch.sum(torch.matmul(output, torch.tensor([[0,1,2,3,4,5,6,7,8,9.]]).transpose(0,1)), dim=1) #output.max(dim=1)[1]
        # indices = output1.max(dim=1)[1].float()

        #print(indices)
        #loss = torch.mean(((indices-target_var.float())*(indices-target_var.float())))
        loss = torch.mean((output1 - target_var)*(output1-target_var)) \
               + torch.mean((output2 - target_var)*(output2-target_var)) \
               + torch.mean((output3 - target_var) * (output3 - target_var))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

def validate(val_loader, model, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = torch.mean((output[0] - target_var)*(output[0]-target_var))
        losses.update(loss)
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, i, len(val_loader), loss=losses))

    return losses.avg

batch_size=12
workers=4
learning_rate=0.1
num_epochs=100
eval_freq=1

nums=4

train_set=DummyDataset(nums, 120)
val_set=DummyDataset(nums, 60)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
model = DummyModel(nums)
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

best_loss=100

for epoch in range(num_epochs):
    train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    if (epoch + 1) % eval_freq == 0:
        loss = validate(val_loader, model, epoch)

        # remember best prec@1 and save checkpoint
        best_loss= min(loss, best_loss)
        print('Best Loss: %.3f\n' % (best_loss))
