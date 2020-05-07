import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.init import normal_, constant_
from torch.nn.utils import clip_grad_norm_
import numpy as np
from ops.utils import AverageMeter

#TODO configs
seq_len = 4
random_seed = 1007

num_epoch = 1000
batch_size = 10
train_num = 1000 * batch_size
val_num = 20 * batch_size
print_freq = 400

learning_rate = 0.01

#TODO random seeds
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def make_a_linear(input_dim, output_dim):
    linear_model = nn.Linear(input_dim, output_dim)
    normal_(linear_model.weight, 0, 0.001)
    constant_(linear_model.bias, 0)
    return linear_model

#TODO models
class SemHash(nn.Module):
    def __init__(self):
        super(SemHash, self).__init__()
        hidden0 = 32
        hidden1 = 16
        self.fc0 = make_a_linear(seq_len, hidden0)
        self.fc1 = make_a_linear(hidden0, hidden1)
        self.fc2 = make_a_linear(hidden1, seq_len)

    def improved_sem_hash(self, v, is_training):
        if is_training:
            noise = torch.normal(0., 1., v.shape)
        else:
            noise = 0
        vn = v + noise
        v1 = torch.relu(torch.clamp_min(1.2 * torch.sigmoid(vn) - 0.1, 1))
        v2 = ((vn < 0).type("torch.FloatTensor") - v1).detach() + v1

        if torch.rand(1, 1).item() > 0.5:
            vd = v1
        else:
            vd = v2
        return vd

    def forward(self, x, is_training=True, plain=False):
        y1 = self.fc0(x)
        y2 = self.fc1(y1)
        v = self.fc2(y2)
        if plain==False:
            attention = self.improved_sem_hash(v, is_training)
            out = torch.sum(attention * x, dim=1)
            return out, attention
        else:
            return torch.relu(v).sum(dim=1), torch.relu(v)

#TODO datasets
#TODO sequence
# 1. each sample has X numbers
# 2. the gt label is the max number minus the min number
#TODO mountains (2,3,4,3,2) => 1
# 1. count the peaks (in neighbours(5), middle==max and middle-side>=2 )
def get_dataset(num_samples):
    #data = torch.randint(-10, 10, (num_samples, seq_len)).type("torch.FloatTensor")
    #labels = data.max(dim=1)[0]+data.min(dim=1)[0]
    data = torch.rand(num_samples, seq_len)
    labels = (((data>0.2) * (data<0.3)).type("torch.FloatTensor")*data).sum(dim=1)
    return data, labels


model = SemHash()
model.train()
train_data, train_gt = get_dataset(train_num)
val_data, val_gt = get_dataset(val_num)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def print_tensor(tensor):
    return "[" + " ".join(["%.4f" % x for x in tensor]) + "]"

#TODO main loop
def train(epoch):
    losses = AverageMeter()
    perm_indices = np.random.permutation(train_num)
    for bi in range(train_num // batch_size):
        indices = perm_indices[bi * batch_size: (bi + 1) * batch_size]
        data = train_data[indices]
        # data = train_data[bi * batch_size: (bi + 1) * batch_size]
        gt = train_gt[indices]
        # gt = train_gt[bi * batch_size: (bi + 1) * batch_size]
        pred, att = model(data, is_training=True, plain=True)
        Goh = torch.zeros(data[0].shape)
        for _i in range(data[0].shape[0]):
            if data[0][_i]>0.2 and data[0][_i]<0.3:
                Goh[_i]=1
        # Goh[data[0].argmax()] = 1
        # Goh[data[0].argmin()] = 1
        loss = torch.mean((pred - gt) * (pred - gt))
        losses.update(loss.item(), data.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if bi % print_freq == 0:
            print("epo-%03d train [%03d|%03d]\tloss:%9.4f(%9.4f)\tseq:%s\tG:%s\tP:%s\tgt:%.2f\tpred:%.2f " %
                  (epoch, bi, train_num // batch_size, losses.val, losses.avg,
                   print_tensor(data[0]), print_tensor(Goh), print_tensor(att[0]), gt[0], pred[0]))
    return losses.avg

def validate(epoch):
    losses = AverageMeter()
    for bi in range(val_num // batch_size):
        data = val_data[bi * batch_size: (bi + 1) * batch_size]
        gt = val_gt[bi * batch_size: (bi + 1) * batch_size]
        pred, att = model(data, is_training=False, plain=True)
        Goh = torch.zeros(data[0].shape)
        for _i in range(data[0].shape[0]):
            if data[0][_i]>0.2 and data[0][_i]<0.3:
                Goh[_i]=1
        # Goh[data[0].argmax()] = 1
        # Goh[data[0].argmin()] = 1
        loss = torch.mean((pred - gt) * (pred - gt))
        losses.update(loss.item(), data.size(0))
        if bi % print_freq ==0:
            print("epo-%03d val   [%03d|%03d]\tloss:%9.4f(%9.4f)\tseq:%s\tG:%s\tP:%s\tgt:%.2f\tpred:%.2f " %
                  (epoch, bi, val_num // batch_size, losses.val, losses.avg,
                   print_tensor(data[0]), print_tensor(Goh), print_tensor(att[0]), gt[0], pred[0]))
    return losses.avg


for i in range(num_epoch):
    train_loss_avg = train(i)
    val_loss_avg = validate(i)
    # val_loss_avg=0
    print("Epoch-%03d, train_loss: %9.4f val_loss: %9.4f"%(i, train_loss_avg, val_loss_avg))
    #print()