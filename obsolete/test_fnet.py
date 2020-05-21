import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from obsolete.ops.dmynet import Conv2dMY
import obsolete.ops.dmynet
from thop import profile
import time

# class DNet(torch.nn.Module):
#     def __init__(self):
#         super(DNet, self).__init__()
#         self.W0 = torch.nn.Parameter(torch.randn(64, 3, 7, 7), requires_grad=True)
#         self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.fc0 = torch.nn.Linear(64, 1000)
#         self.fc1 = torch.nn.Linear(16, 1000)
#
#     def forward(self, input, **kwargs):
#         signal = kwargs["signal"]
#         if signal == 0:
#             x = torch.nn.functional.conv2d(input, self.W0, None, 1)
#             fc=self.fc0
#         else:
#             x = torch.nn.functional.conv2d(input, self.W0[:16], None, 1)
#             fc = self.fc1
#         return fc(torch.flatten(self.avgpool(x), 1))
#
# model = DNet()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# model.train()
# for i in range(42):
#     pred = model(torch.randn(1, 3, 224, 224), signal=i % 2)
#     loss = torch.nn.CrossEntropyLoss()(pred, torch.tensor([666]))
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     print("iter:%04d pred:%d loss:%.4f" % (i, torch.argmax(pred.detach()), loss.item()))
# exit()
class SingleNet(nn.Module):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.conv0 = Conv2dMY(3, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv1 = Conv2dMY(64, 128, kernel_size=3, stride=1, bias=False, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(128, 1000)
        self.fc1 = nn.Linear(32, 1000)

    def forward(self, x, **kwargs):
        signal = kwargs["signal"]
        if signal == 0:
            x=self.conv0(x)
            x=self.conv1(x)
            x=self.avgpool(x)
            x=torch.flatten(x,1)
            return self.fc0(x)
        else:
            x = self.conv0(x, in_begin=0,in_end=3,out_begin=0,out_end=16)
            x = self.conv1(x, in_begin=0,in_end=16,out_begin=0,out_end=32)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc1(x)

class DoubleFNet(nn.Module):
    def __init__(self):
        super(DoubleFNet, self).__init__()
        self.W0 = nn.Parameter(torch.randn(64,   3, 3, 3), requires_grad=True)
        self.W1 = nn.Parameter(torch.randn(128, 64, 3, 3), requires_grad=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc0 = nn.Linear(128, 1000)
        self.fc1 = nn.Linear(32, 1000)

    def forward(self, input, **kwargs):
        signal = kwargs["signal"]
        if signal == 0:
            x = F.conv2d(input, self.W0, None, 1)
            x = F.conv2d(x, self.W1, None, 1)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc0(x)
        else:
            x = F.conv2d(input, self.W0[:16], None, 1)
            x = F.conv2d(x, self.W1[:32,:16], None, 1)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc1(x)

resolution=224
dummy_data = torch.randn(1, 3, resolution, resolution)

def timing(model, s):
    flops, params = profile(model, inputs=(dummy_data,), custom_ops={
        obsolete.ops.dmynet.Conv2dMY: obsolete.ops.dmynet.count_conv_my})
    print(flops / 1e9, params / 1e6)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.train()

    batchsize=64
    num_iter=1000

    _data = torch.randn(batchsize, 3, resolution, resolution)
    t0=time.time()
    for i in range(num_iter):
        pred = model(_data.cuda())
        label = torch.tensor([999]*batchsize)
        loss = criterion(pred, label.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print("iter:%04d pred:%s loss:%.4f" % (i, torch.argmax(pred.cpu().detach()), loss.cpu().item()))
    t1=time.time()
    print(s,":",t1-t0,"secs")
    print()

# model = ops.fnet.FNet("fresnet50", [64, 64, 16, 8], True, None)
# model = DoubleFNet()
# model = SingleNet()
model = obsolete.ops.dmynet.dmynet50(num_filters_list=[64, 32, 16, 8], default_signal=0)
# flops, params = profile(model, inputs=(dummy_data,))
# print(flops / 1e9, params / 1e6)
#
# model = getattr(torchvision.models, "resnet50")(False)

timing(
    obsolete.ops.dmynet.dmynet50(num_filters_list=[64, 32, 16, 8], default_signal=0),
    "dmynet50, signal=0")

timing(
    obsolete.ops.dmynet.dmynet50(num_filters_list=[64, 32, 16, 8], default_signal=1),
    "dmynet50, signal=1")

timing(
    obsolete.ops.dmynet.dmynet50(num_filters_list=[64, 32, 16, 8], default_signal=2),
    "dmynet50, signal=2")

timing(
    obsolete.ops.dmynet.dmynet50(num_filters_list=[64, 32, 16, 8], default_signal=3),
    "dmynet50, signal=3")

timing(
    getattr(torchvision.models, "resnet50")(False),
    "resnet50           ")

exit()


criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.1)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model.train()

for i in range(20):
    pred=model(dummy_data.cuda(), signal=i%4)
    label = torch.tensor([999])
    loss = criterion(pred, label.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("iter:%04d pred:%s loss:%.4f"%(i, torch.argmax(pred.cpu().detach()), loss.cpu().item()))
    # print(model.module.conv1.weight.cpu().detach().numpy()[31:33,15:17,0,0])
    # print(model.module.base_model_list[0].layer3[5].conv3.weight.cpu().detach().numpy()[:5, 0, 0, 0])
    # print(model.module.base_model_list[1].layer3[5].conv3.weight.cpu().detach().numpy()[:5, 0, 0, 0])
    # print(model.module.base_model_list[2].layer3[5].conv3.weight.cpu().detach().numpy()[:5, 0, 0, 0])
    # print(model.module.base_model_list[3].layer3[5].conv3.weight.cpu().detach().numpy()[:5, 0, 0, 0])
    #print(model.module.W0.cpu().detach().numpy()[:1, :4,0,0])
    # print(model.module.W1.cpu().detach().numpy()[31:33, 14:14+4,0,0])