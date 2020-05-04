# TODO modified version of https://github.com/pytorch/vision/blob/master/torchvision/models/utils.py
# TODO Yue Meng
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['DHSNet', 'dhsnet18', 'dhsnet34', 'dhsnet50', 'dhsnet101', 'dhsnet152']


model_urls = {
    'dhsnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'dhsnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'dhsnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'dhsnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'dhsnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

#TODO
# 1. dynamic conv weights similar as dmynet
# 2. caches for different segment of input channel-outputted features
# 3. channel signal, common/fusion conv op, and reset op
class Conv2dHS(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 dhs_enabled=True, num_filters_list=[], default_signal=None, args=None):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride,stride)
        padding = (padding,padding)
        dilation = (dilation,dilation)
        super(Conv2dHS, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        if args.dhs_print_net_states:
            print("conv: enabled:%s"%dhs_enabled)
        self.dhs_enabled = dhs_enabled
        self.num_filters_list = num_filters_list  #TODO: e.g. 64, 48, 32, 16 => [16, 32], [32, 48], [48, 64]
        self.default_signal = default_signal
        self.args=args

        # self.cache_list = [None for _ in range(len(self.num_filters_list)-1)]

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

    def forward(self, x, history, signal, mode_op):
        # TODO mode_op = 0,1,2,3
        # TODO mode_op = 0: normal conv, first image level
        # TODO mode_op = 1: normal conv, rest levels
        # TODO mode_op = 2: dynamic conv(type=0), reset the caches (for first frame)
        # TODO mode_op = 3: dynamic conv(type=0), cache and change input only
        # TODO mode_op = 4: dynamic conv(type=1), reset the caches (for first frame)
        # TODO mode_op = 5: dynamic conv(type=1), change both input and output

        if self.dhs_enabled == False: #TODO normal conv
            output = self._conv_forward(x, self.weight, dilation=1)  # TODO no shape problem

            if self.args.dynamic_channel:
                if mode_op == 2:
                    self.cmask_book = torch.zeros(1, len(self.num_filters_list), output.shape[1]).to(output.device)
                    for i, num_filter in enumerate(self.num_filters_list):
                        self.cmask_book[0, i, :output.shape[1] * num_filter // 64] = 1.
                output_mask = torch.sum(self.cmask_book * signal.unsqueeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)
                if not self.args.no_pre_bn_mask:
                    output = output * output_mask
            else:
                output_mask = None
            return output, None, output_mask

        if mode_op == 0 or mode_op == 1:
            exit("We don't design mode_op={0,1}")
        elif mode_op == 2:  # TODO cache saves input_channels
            if self.args.dhs_fix_history_ratio:
                input_mask = torch.zeros(1, x.shape[1], 1, 1).to(x.device)
                input_mask[:x.shape[1] * self.args.dhs_current_ratio// 64] = 1.  # TODO upper for current (one)
            elif self.args.dhs_fuse_history:
                input_mask = 1
            else:
                self.mask_book = torch.zeros(1, len(self.num_filters_list), x.shape[1]).to(x.device)
                for i, num_filter in enumerate(self.num_filters_list):
                    self.mask_book[0, i, :x.shape[1] * num_filter // 64] = 1.
                # TODO mask_book  shape (1, K, C)
                # TODO signal     shape (B, K, 1)
                # TODO input_mask shape (B, C) => (B, C, 1, 1)
                # TODO output_mask shape (B, C') => (B, C', 1, 1)
                input_mask = torch.sum(self.mask_book * signal.unsqueeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)
            real_input = x * input_mask
            output = self._conv_forward(real_input, self.weight, dilation=1)  # TODO no shape problem
            if self.args.dhs_no_history:
                new_history = None
            elif self.args.dhs_rand_history:
                new_history = torch.rand(x.shape).to(x.device)
            elif self.args.dhs_history_no_grad:
                new_history = real_input.detach()
            elif self.args.dhs_fix_history_ratio:
                new_history = x
            elif self.args.dhs_fuse_history:
                new_history = x
            else:
                new_history = real_input

            if self.args.dynamic_channel:
                self.cmask_book = torch.zeros(1, len(self.num_filters_list), output.shape[1]).to(output.device)
                for i, num_filter in enumerate(self.num_filters_list):
                    self.cmask_book[0, i, :output.shape[1] * num_filter // 64] = 1.
                output_mask = torch.sum(self.cmask_book * signal.unsqueeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)
                if not self.args.no_pre_bn_mask:
                    output = output * output_mask
            else:
                output_mask = None
            return output, new_history, output_mask

        elif mode_op == 3:  # TODO cache saves input_channels
            #TODO 1. fuse history/padding with current input features
            if self.args.dhs_fix_history_ratio:
                input_mask = torch.zeros(1, x.shape[1], 1, 1).to(x.device)
                input_mask[:x.shape[1] * self.args.dhs_current_ratio// 64] = 1.  # TODO upper for current (one)
            elif self.args.dhs_fuse_history:
                input_mask = None
            else:
                input_mask = torch.sum(self.mask_book * signal.unsqueeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)
            if any([x is None for x in [self.args.dhs_no_history, self.args.dhs_one_history, self.args.dhs_rand_history, self.args.dhs_history_no_grad]]):
                print("Detected", [self.args.dhs_no_history, self.args.dhs_one_history, self.args.dhs_rand_history, self.args.dhs_history_no_grad])
                exit()

            if self.args.dhs_no_history:
                real_input = x * input_mask
                new_history = None
            elif self.args.dhs_one_history:
                new_history = x * input_mask
                real_input = new_history + history * (1 - input_mask)
            elif self.args.dhs_rand_history:
                real_input = x * input_mask + history * (1 - input_mask)
                new_history = history
            elif self.args.dhs_history_no_grad:
                real_input = x * input_mask + history * (1 - input_mask)
                new_history = real_input.detach()
            elif self.args.dhs_fix_history_ratio:
                real_input = x * input_mask + history * (1 - input_mask)
                new_history = x
            elif self.args.dhs_fuse_history:
                real_input = (x + history) / 2
                new_history = x
            else:
                real_input = x * input_mask + history * (1 - input_mask)
                new_history = real_input
            #TODO 2. dynamic channel conv
            #TODO 3. get new history
            output = self._conv_forward(real_input, self.weight, dilation=1)
            if self.args.dynamic_channel:
                output_mask = torch.sum(self.cmask_book * signal.unsqueeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)
                if not self.args.no_pre_bn_mask:
                    output = output * output_mask
            else:
                output_mask = None
            return output, new_history, output_mask

        elif mode_op == 4 or mode_op == 5: # TODO cache saves output channels in sections
            curr_output_list=[]
            output = torch.tensor(0.0)
            #TODO 1 current stage
            for _ in range(len(self.cache_list)):
                curr_output_list.append(self._conv_forward())
                output = output + curr_output_list[-1]

            #TODO 2 history fusion if not for first frame
            if mode_op == 5:
                num_cache = signal + 1
                # for i in range(num_cache):
                #     output = curr_output_list + self.cache_list[i]

            #TODO 3 history fusion add
            # for i in range(len(self.cache_list)):
                # self.cache_list[i] = curr_output_list[i]

            return output, None, None



def count_conv_hs(m: Conv2dHS, x, y):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0
    # ratio_cnts = torch.sum(x[2],dim=0)
    # channel_ratio=0
    # for i,num_filters in enumerate(m.num_filters_list):
    #     channel_ratio += ratio_cnts[i]*num_filters /64
    # channel_ratio = channel_ratio / x[2].shape[0]
    if m.args.dynamic_channel:
        channel_ratio = m.num_filters_list[m.default_signal] / 64
    else:
        channel_ratio = 1
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    tensor_ops = x[0].nelement() * 3 #TODO two point-wise product and one sum-up
    conv_ops = y[0].nelement() * (x[0].shape[1] // m.groups * kernel_ops + bias_ops) * channel_ratio
    total_ops = tensor_ops + conv_ops

    m.total_ops += torch.Tensor([int(total_ops)])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dhs_enabled=True, num_filters_list=[],
            default_signal=None, args=None):
    """3x3 convolution with padding"""
    return Conv2dHS(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     dhs_enabled=dhs_enabled, num_filters_list=num_filters_list,
                    default_signal=default_signal, args=args)


def conv1x1(in_planes, out_planes, stride=1, dhs_enabled=True, num_filters_list=[], default_signal=None, args=None):
    """1x1 convolution"""
    return Conv2dHS(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                    dhs_enabled=dhs_enabled, num_filters_list=num_filters_list,
                    default_signal=default_signal, args=args)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_filters_list=None, default_signal=0, args=None,
                 is_first_block=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        #TODO(yue) Both self.conv1 and self.downsample layers downsample the input when stride != 1
        shall_first = True #TODO holds for all_level or block_level
        if args.dhs_stage_level:
            shall_first = is_first_block #TODO otherwise judge whether it's the first block

        self.conv1 = conv3x3(inplanes, planes, stride, dhs_enabled=shall_first, num_filters_list=num_filters_list,
                             default_signal=default_signal, args=args)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dhs_enabled=args.dhs_all_level, num_filters_list=num_filters_list,
                             default_signal=default_signal, args=args)
        self.bn2 = norm_layer(planes)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.num_filters_list = num_filters_list
        self.stride = stride
        self.default_signal = default_signal
        self.args = args


    def forward(self, x, **kwargs):
        conv_signal = self.default_signal
        if "signal" in kwargs:
            conv_signal = kwargs["signal"]

        history = kwargs["history"]
        if history is None:
            history = [None, None, None]
        mode_op = kwargs["mode_op"]

        identity = x

        out, nh0, c0 = self.conv1(x, history=history[0], signal=conv_signal, mode_op=mode_op)
        out = self.bn1(out)
        if self.args.bn_channel_mask:
            out = out * c0
        out = self.relu(out)

        out, nh1, c1 = self.conv2(out, history=history[1], signal=conv_signal, mode_op=mode_op)
        out = self.bn2(out)
        if self.args.bn_channel_mask:
            out = out * c1

        if self.downsample0 is not None:
            y, nh2, c2 = self.downsample0(x, history=history[2], signal=conv_signal, mode_op=mode_op)
            identity = self.downsample1(y)
            if self.args.bn_channel_mask:
                identity = identity * c2
        else:
            nh2 = None
        out += identity
        out = self.relu(out)

        return out, [nh0, nh1, nh2]


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_filters_list = None, default_signal=0,
                 args=None, is_first_block=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # TODO(yue) Both self.conv1 and self.downsample layers downsample the input when stride != 1
        shall_first = True  # TODO holds for all_level or block_level
        if args.dhs_stage_level:
            shall_first = is_first_block  # TODO otherwise judge whether it's the first block
        self.conv1 = conv1x1(inplanes, width, dhs_enabled=shall_first, num_filters_list=num_filters_list,
                             default_signal=default_signal, args=args)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, dhs_enabled=args.dhs_all_level,
                             num_filters_list=num_filters_list, default_signal=default_signal, args=args)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, dhs_enabled=args.dhs_all_level,
                             num_filters_list=num_filters_list, default_signal=default_signal, args=args)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride
        self.num_filters_list=num_filters_list
        # self.bn1 = norm_layer(width)
        # self.bn2 = norm_layer(width)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.default_signal = default_signal
        self.args=args

    def forward(self, x, **kwargs):
        # print("forward", self.last_conv_same, self.downsample0 is not None)
        conv_signal = self.default_signal
        if "signal" in kwargs:
            conv_signal = kwargs["signal"]

        history = kwargs["history"]
        if history is None:
            history = [None, None, None, None]
        mode_op = kwargs["mode_op"]

        identity = x

        out, nh0, c0 = self.conv1(x, history=history[0], signal=conv_signal, mode_op=mode_op)
        out = self.bn1(out)
        if self.args.bn_channel_mask:
            out = out * c0
        out = self.relu(out)

        out, nh1, c1  = self.conv2(out, history=history[1], signal=conv_signal, mode_op=mode_op)
        out = self.bn2(out)
        if self.args.bn_channel_mask:
            out = out * c1
        out = self.relu(out)

        out, nh2, c2 = self.conv3(out, history=history[2], signal=conv_signal, mode_op=mode_op)

        out = self.bn3(out)
        if self.args.bn_channel_mask:
            out = out * c2

        if self.downsample0 is not None:
            y, nh3, c3 = self.downsample0(x, history=history[3], signal=conv_signal, mode_op=mode_op)
            identity = self.downsample1(y)
            if self.args.bn_channel_mask:
                identity = identity * c3
        else:
            nh3 = None
        out += identity
        out = self.relu(out)

        return out, [nh0, nh1, nh2, nh3]


class DHSNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_filters_list=[64, 32, 16, 8], default_signal=0,
                 args=None):
        super(DHSNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_filters_list=num_filters_list
        self.diversity=len(num_filters_list)
        self.inplanes = num_filters_list[0]
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if args.dhs_print_net_states:
            print("NETWORK")

        self.conv1 = Conv2dHS(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                              bias=False, dhs_enabled=False, num_filters_list=num_filters_list,
                              default_signal=default_signal, args=args)
        self.bn1 = norm_layer(self.inplanes)
        self.default_signal = default_signal
        self.args = args

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * 1, layers[0])
        self.layer2 = self._make_layer(block, 64 * 2, layers[1],
                                       stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64 * 4, layers[2],
                                       stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 64 * 8, layers[3],
                                       stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if args.dhs_print_net_states:
            print("")

        for m in self.modules():
            if isinstance(m, Conv2dHS):
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
                    # for bn in m.bn2s:
                    #     nn.init.constant_(bn.weight, 0)
                    nn.init.constant_(m.bn2.weight,0)

    def _make_layer(self, block, planes_list_0, blocks, stride=1, dilate=False):
        if self.args.dhs_print_net_states:
            print("layer")
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1 = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list_0 * block.expansion:
            shall_enable = self.args.dhs_all_level
            downsample0 = conv1x1(self.inplanes, planes_list_0 * block.expansion, stride,
                                  dhs_enabled=shall_enable, num_filters_list=self.num_filters_list,
                                  default_signal=self.default_signal, args=self.args)
            downsample1 = norm_layer(planes_list_0 * block.expansion)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list_0, stride, downsample0, downsample1, self.groups,
                            self.base_width, previous_dilation, norm_layer, num_filters_list=self.num_filters_list,
                            default_signal=self.default_signal, args=self.args, is_first_block=True))
        self.inplanes = planes_list_0 * block.expansion
        for k in range(1, blocks):
            layers.append(block(self.inplanes, planes_list_0, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_filters_list=self.num_filters_list,
                                default_signal=self.default_signal, args=self.args, is_first_block=False))
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

    def forward(self, input_data, **kwargs):

        #TODO x.shape
        _B, _T, _C, _H, _W = input_data.shape

        signal = torch.ones(_B, _T).to(input_data.device) * self.default_signal
        if "signal" in kwargs:
            signal = kwargs["signal"]

        out_list = []
        mode_op = 2

        #TODO history is block and conv-level state
        history_rack = []
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            history_rack.append([])
            for _ in layers:
                history_rack[-1].append(None)

        for t in range(_T):
            x, null_history, c0 = self.conv1(input_data[:, t], history=None, signal=signal[:, t], mode_op=mode_op)
            x = self.bn1(x)
            if self.args.bn_channel_mask:
                x = x * c0
            x = self.relu(x)
            x = self.maxpool(x)

            for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for bi, block in enumerate(layers):
                    x, new_history = block(x, history=history_rack[li][bi], signal=signal[:, t], mode_op=mode_op)
                    history_rack[li][bi] = new_history
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)
            out_list.append(out)
            if t == 0:
                mode_op += 1
        return torch.stack(out_list, dim=1)


def _dhsnet(arch, block, layers, pretrained, progress, **kwargs):
    model = DHSNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
        #     #TODO batchnorm
        #     if "bn" in k:
        #         # TODO layer1.0.bn1.weight-> layer1.0.bn1s.0.weight
        #         before_bn,after_bn = k.split("bn")
        #         for fc_i in range(1, len(kwargs["num_filters_list"])):
        #             if kwargs["last_conv_same"] and "layer4.2.bn3" in k:
        #                 break
        #             origin_len = pretrained_dict[k].shape[0]
        #             slicing_len = origin_len * kwargs["num_filters_list"][fc_i]// 64
        #             kvs_to_add.append((before_bn+"bn"+after_bn[0]+"s.%d"%(fc_i)+after_bn[1:], pretrained_dict[k][:slicing_len]))
        #         old_to_new_pairs.append((k, before_bn+"bn"+after_bn[0]+"s.0"+after_bn[1:]))
            # TODO downsample
            if "downsample.0" in k:
                # TODO layer1.0.downsample.0.weight-> layer1.0.downsample0.weight
                old_to_new_pairs.append((k, k.replace("downsample.0",  "downsample0")))
            # TODO downsample
            elif "downsample.1" in k:
                # TODO layer4.0.downsample.1.weight -> layer4.0.downsample1s.0.weight
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))
        #     # TODO fc layers
        #     elif "fc" in k:
        #         # TODO fc.weight -> fcs.0.weight
        #         old_to_new_pairs.append((k, k.replace("fc", "fcs.0")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def dhsnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhsnet('dhsnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def dhsnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhsnet('dhsnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def dhsnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhsnet('dhsnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def dhsnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhsnet('dhsnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def dhsnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhsnet('dhsnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)