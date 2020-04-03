# TODO modified version of https://github.com/pytorch/vision/blob/master/torchvision/models/utils.py
# TODO Yue Meng
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['DMyNet', 'dmynet18', 'dmynet34', 'dmynet50', 'dmynet101',
           'dmynet152', 'dmynext50_32x4d', 'dmynext101_32x8d',
           'wide_dmynet50_2', 'wide_dmynet101_2']


model_urls = {
    'dmynet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'dmynet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'dmynet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'dmynet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'dmynet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'dmynext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'dmynext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_dmynet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_dmynet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Conv2dMY(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride,stride)
        padding = (padding,padding)
        dilation = (dilation,dilation)
        super(Conv2dMY, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)


    def _conv_forward(self, x, weight, dilation):

        kernel_size=weight.shape[2]
        d_padding = ((kernel_size-1)//2) * dilation

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            (0,0), dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        (d_padding, d_padding), dilation, self.groups)
        # return F.conv2d(x, weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)

    def forward(self, x, in_begin=0, in_end=64, out_begin=0, out_end=64, dilation=1):
        in_begin = self.weight.shape[1] * in_begin // 64
        in_end = self.weight.shape[1] * in_end // 64
        out_begin = self.weight.shape[0] * out_begin // 64
        out_end = self.weight.shape[0] * out_end // 64
        return self._conv_forward(x, self.weight[out_begin:out_end, in_begin:in_end], dilation)


def count_conv_my(m: Conv2dMY, x, y):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (x[0].shape[1] // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2dMY(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dMY(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1s=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_filters_list = None, default_signal=0,
                 last_conv_same=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = norm_layer(planes)
        self.downsample0 = downsample0
        self.downsample1s = downsample1s
        self.last_conv_same = last_conv_same
        self.num_filters_list = num_filters_list
        self.stride = stride
        self.bn1s = torch.nn.ModuleList()
        self.bn2s = torch.nn.ModuleList()
        for bns in [self.bn1s, self.bn2s]:
            for num_filters in num_filters_list:
                bns.append(norm_layer(planes * num_filters // 64))
                if bns ==self.bn2s and self.last_conv_same:
                    break
        self.default_signal = default_signal

    def forward(self, x, **kwargs):
        conv_signal = self.default_signal % 10
        dila_signal = (self.default_signal // 10) % 10 + 1
        if "signal" in kwargs:
            conv_signal = kwargs["signal"] % 10
            dila_signal = (kwargs["signal"] // 10) % 10 + 1
        in_begin = 0
        in_end = self.num_filters_list[conv_signal]
        out_begin = 0
        out_end = self.num_filters_list[conv_signal]

        identity = x

        out = self.conv1(x, in_begin, in_end, out_begin, out_end, dila_signal)
        out = self.bn1s[conv_signal](out)
        out = self.relu(out)

        last_in_begin = in_begin
        last_out_begin = out_begin
        if self.last_conv_same:
            last_in_end = in_end
            last_out_end = self.num_filters_list[0]
            last_dila_signal = 1
            last_conv_signal = 0
        else:
            last_in_end = in_end
            last_out_end = out_end
            last_dila_signal = dila_signal
            last_conv_signal = conv_signal

        out = self.conv2(out, last_in_begin, last_in_end, last_out_begin, last_out_end, last_dila_signal)
        out = self.bn2s[last_conv_signal](out)

        if self.downsample0 is not None:
            y = self.downsample0(x, in_begin, in_end, out_begin, out_end)
            identity = self.downsample1s[conv_signal](y)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1s=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_filters_list = None, default_signal=0,
                 last_conv_same=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        #self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        #self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample0 = downsample0
        self.downsample1s = downsample1s
        self.stride = stride
        self.last_conv_same = last_conv_same
        self.num_filters_list=num_filters_list
        self.bn1s = torch.nn.ModuleList()
        self.bn2s = torch.nn.ModuleList()
        self.bn3s = torch.nn.ModuleList()
        for bns, _width in [(self.bn1s,width), (self.bn2s,width), (self.bn3s,planes * self.expansion)]:
            for num_filters in num_filters_list:
                bns.append(norm_layer(_width * num_filters // 64))
                if bns ==self.bn3s and last_conv_same:
                    break
        self.default_signal = default_signal

    def forward(self, x, **kwargs):
        # print("forward", self.last_conv_same, self.downsample0 is not None)
        conv_signal = self.default_signal % 10
        dila_signal = (self.default_signal // 10) % 10 + 1
        if "signal" in kwargs:
            conv_signal = kwargs["signal"] % 10
            dila_signal = (kwargs["signal"] // 10) % 10 + 1
        in_begin = 0
        in_end = self.num_filters_list[conv_signal]
        out_begin = 0
        out_end = self.num_filters_list[conv_signal]

        identity = x

        out = self.conv1(x, in_begin, in_end, out_begin, out_end, dila_signal)
        out = self.bn1s[conv_signal](out)
        out = self.relu(out)

        out = self.conv2(out, in_begin, in_end, out_begin, out_end, dila_signal)
        out = self.bn2s[conv_signal](out)
        out = self.relu(out)

        last_in_begin = in_begin
        last_out_begin = out_begin
        if self.last_conv_same:
            last_in_end = in_end
            last_out_end = self.num_filters_list[0]
            last_dila_signal = 1
            last_conv_signal = 0
        else:
            last_in_end = in_end
            last_out_end = out_end
            last_dila_signal = dila_signal
            last_conv_signal = conv_signal
        # print("bottleneck:",out.shape, last_in_begin, last_in_end, last_out_begin,last_out_end,last_dila_signal)
        out = self.conv3(out, last_in_begin, last_in_end, last_out_begin, last_out_end, last_dila_signal)
        out = self.bn3s[last_conv_signal](out)

        if self.downsample0 is not None:
            y = self.downsample0(x, in_begin, in_end, out_begin, out_end)
            identity = self.downsample1s[conv_signal](y)
        out += identity
        out = self.relu(out)

        return out


class DMyNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_filters_list=[64, 32, 16, 8], default_signal=0, last_conv_same=False):
        super(DMyNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_filters_list=num_filters_list
        self.diversity=len(num_filters_list)
        self.inplanes = num_filters_list[0]
        self.dilation = 1
        self.last_conv_same = last_conv_same

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2dMY(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1s=torch.nn.ModuleList()
        for num_filters in num_filters_list:
            self.bn1s.append(norm_layer(num_filters))
        self.default_signal = default_signal

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, [num_filters * 1 for num_filters in num_filters_list], layers[0])
        self.layer2 = self._make_layer(block, [num_filters * 2 for num_filters in num_filters_list], layers[1],
                                       stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, [num_filters * 4 for num_filters in num_filters_list], layers[2],
                                       stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, [num_filters * 8 for num_filters in num_filters_list], layers[3],
                                       stride=2, dilate=replace_stride_with_dilation[2], last_conv_same=last_conv_same)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = torch.nn.ModuleList()
        for num_filters in num_filters_list:
            self.fcs.append(nn.Linear(num_filters * 8 * block.expansion, num_classes))
            if last_conv_same:
                break


        for m in self.modules():
            if isinstance(m, Conv2dMY):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    for bn in m.bn3s:
                        nn.init.constant_(bn.weight, 0)
                elif isinstance(m, BasicBlock):
                    for bn in m.bn2s:
                        nn.init.constant_(bn.weight, 0)

    def _make_layer(self, block, planes_list, blocks, stride=1, dilate=False, last_conv_same=False):
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1s = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list[0] * block.expansion:
            downsample0 = conv1x1(self.inplanes, planes_list[0] * block.expansion, stride)
            downsample1s = torch.nn.ModuleList()
            for plane in planes_list:
                downsample1s.append(norm_layer(plane * block.expansion))

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list[0], stride, downsample0, downsample1s, self.groups,
                            self.base_width, previous_dilation, norm_layer, num_filters_list = self.num_filters_list,
                            default_signal = self.default_signal))
        self.inplanes = planes_list[0] * block.expansion
        for k in range(1, blocks):
            layers.append(block(self.inplanes, planes_list[0], groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_filters_list = self.num_filters_list,
                                default_signal = self.default_signal, last_conv_same = last_conv_same and k == blocks-1))

        return layers

    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #
    #     return x

    def forward(self, x, **kwargs):
        conv_signal = self.default_signal % 10
        dila_signal = (self.default_signal // 10) % 10 + 1
        if "signal" in kwargs:
            conv_signal = kwargs["signal"] % 10
            dila_signal = (kwargs["signal"] // 10) % 10 + 1

        in_begin = 0
        in_end = 64  #TODO initial 3 channels ~ 64/64
        out_begin = 0
        out_end = self.num_filters_list[conv_signal]

        x = self.conv1(x, in_begin, in_end, out_begin, out_end, dila_signal)
        x = self.bn1s[conv_signal](x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layers:
                x = block(x, signal=(dila_signal-1)*10+conv_signal)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.last_conv_same:
            last_conv_signal = 0
        else:
            last_conv_signal = conv_signal
        x = self.fcs[last_conv_signal](x)

        return x


def _dmynet(arch, block, layers, pretrained, progress, **kwargs):
    model = DMyNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
            #TODO batchnorm
            if "bn" in k:
                # TODO layer1.0.bn1.weight-> layer1.0.bn1s.0.weight
                before_bn,after_bn = k.split("bn")
                for fc_i in range(1, len(kwargs["num_filters_list"])):
                    if kwargs["last_conv_same"] and "layer4.2.bn3" in k:
                        break
                    origin_len = pretrained_dict[k].shape[0]
                    slicing_len = origin_len * kwargs["num_filters_list"][fc_i]// 64
                    kvs_to_add.append((before_bn+"bn"+after_bn[0]+"s.%d"%(fc_i)+after_bn[1:], pretrained_dict[k][:slicing_len]))
                old_to_new_pairs.append((k, before_bn+"bn"+after_bn[0]+"s.0"+after_bn[1:]))
            # TODO downsample
            elif "downsample.0" in k:
                # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                old_to_new_pairs.append((k, k.replace("downsample.0",  "downsample0")))
            # TODO downsample
            elif "downsample.1" in k:
                # TODO layer4.0.downsample.1.weight -> layer4.0.downsample1s.0.weight
                for fc_i in range(1, len(kwargs["num_filters_list"])):
                    origin_len = pretrained_dict[k].shape[0]
                    slicing_len = origin_len * kwargs["num_filters_list"][fc_i]// 64
                    kvs_to_add.append((k.replace("downsample.1", "downsample1s.%d"%(fc_i)), pretrained_dict[k][:slicing_len]))
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1s.0")))
            # TODO fc layers
            elif "fc" in k:
                # TODO fc.weight -> fcs.0.weight
                old_to_new_pairs.append((k, k.replace("fc", "fcs.0")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def dmynet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dmynet('dmynet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def dmynet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dmynet('dmynet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def dmynet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dmynet('dmynet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def dmynet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dmynet('dmynet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def dmynet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dmynet('dmynet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def dmynext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _dmynet('dmynext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def dmynext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _dmynet('dmynext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_dmynet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _dmynet('wide_dmynet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_dmynet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _dmynet('wide_dmynet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)