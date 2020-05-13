# TODO modified version of https://github.com/pytorch/vision/blob/master/torchvision/models/utils.py
# TODO Yue Meng
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

from ops.utils import count_relu_flops, count_bn_flops, count_fc_flops, count_conv2d_flops

__all__ = ['GateNet', 'gatenet18', 'gatenet34', 'gatenet50', 'gatenet101', 'gatenet152']

model_urls = {
    'gatenet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'gatenet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'gatenet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'gatenet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'gatenet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def list_sum(obj):
    if isinstance(obj, list):
        if len(obj)==0:
            return 0
        else:
            return sum(list_sum(x) for x in obj)
    else:
        return obj


def gumbel_sigmoid(logits, tau=1, hard=False):
    U = torch.empty_like(logits).uniform_()
    L = torch.log(U) - torch.log(1 - U)  # L(U)=logU−log(1−U)
    gumbels = (logits + L) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        # index = y_soft.max(dim, keepdim=True)[1]
        # y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        y_hard = torch.zeros_like(y_soft)
        y_hard[torch.where(y_soft > 0.5)] = 1
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

# https://github.com/tensorflow/tensor2tensor/blob/a9da9635917814af890a31a060c5b29d31b2f906/
# tensor2tensor/layers/common_layers.py#L161
def inverse_exp_decay(max_step, min_value = 0.01, curr_step = None):
    inv_base = torch.exp(torch.log(torch.tensor(min_value))/max_step)
    return (inv_base ** max(max_step - curr_step, 0.0)).item()

# https://github.com/tensorflow/tensor2tensor/blob/a9da9635917814af890a31a060c5b29d31b2f906/
# tensor2tensor/layers/common_layers.py#L143
def saturating_sigmoid(x):
    # y = tf.sigmoid(x)
    # return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))
    y = torch.sigmoid(x)
    return torch.clamp(1.2 * y - 0.1, 0, 1)


def improved_sem_hash(logits, is_training, sem_hash_dense_random, max_step, curr_step):
    noise = torch.normal(0., 1., logits.shape).to(logits.device) if is_training else 0
    vn = logits + noise
    v = saturating_sigmoid(vn)
    # v1 = torch.relu(torch.clamp_min(1.2 * torch.sigmoid(vn) - 0.1, 1))
    v_discrete = ((vn < 0).type(v.type()) - v).detach() + v

    if sem_hash_dense_random:
        mask = torch.randint(0, 2, v.shape).type(v.type()).to(v.device) if is_training else 0
        return mask * v + (1-mask) * v_discrete
    else:
        # if is_training and torch.rand(1, 1).item() > 0.5:
        #     return v1
        # else:
        #     return ((vn < 0).type(v1.type()) - v1).detach() + v1
        if is_training:
            threshold = inverse_exp_decay(max_step=max_step, curr_step=curr_step)
        else:
            threshold = 1
        rand = torch.rand_like(v)
        return torch.where(rand < threshold, v_discrete, v)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        if self.args.gate_local_policy:

            if self.args.gate_tanh:
                self.tanh = nn.Tanh()
                self.relu_not_inplace = nn.ReLU(inplace=False)
            elif self.args.gate_sigmoid:
                self.sigmoid = nn.Sigmoid()

            in_factor = 1
            out_factor = 2 if any([self.args.gate_gumbel_softmax, self.args.gate_tanh, self.args.gate_sigmoid]) else 1
            if self.args.gate_history:
                in_factor = 2 if self.args.fusion_type == "cat" else 1
                if self.args.gate_gumbel_softmax or self.args.gate_tanh or self.args.gate_sigmoid:
                    out_factor = 3
                elif self.args.gate_sem_hash:
                    out_factor = 2

                if self.args.gate_history_conv_type in ['conv1x1','conv1x1bnrelu']:
                    self.gate_hist_conv = conv1x1(planes, planes)
                    if self.args.gate_history_conv_type == 'conv1x1bnrelu':
                        self.gate_hist_bnrelu = nn.Sequential(norm_layer(planes), nn.ReLU(inplace=True))
                if self.args.gate_history_conv_type == 'conv1x1_list':
                    self.gate_hist_conv = torch.nn.ModuleList()
                    for _t in range(self.args.num_segments):
                        self.gate_hist_conv.append(conv1x1(planes, planes))
                if self.args.gate_history_conv_type == 'conv1x1_res':
                    self.gate_hist_conv = conv1x1(planes, planes)

            self.gate_fc0 = nn.Linear(inplanes * in_factor, self.args.gate_hidden_dim)

            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(self.args.gate_hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            self.gate_fc1 = nn.Linear(self.args.gate_hidden_dim, planes * out_factor)
            self.num_channels = planes
            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)

    def count_flops(self, input_data_shape, **kwargs):
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        if self.args.gate_history_conv_type in ['conv1x1','conv1x1bnrelu']:
            gate_history_conv_flops, history_conv_shape = count_conv2d_flops(conv2_out_shape, self.gate_hist_conv)
            if self.args.gate_history_conv_type == 'conv1x1bnrelu':
                gate_history_conv_flops += count_bn_flops(history_conv_shape)[0] + count_relu_flops(history_conv_shape)[0]
        elif self.args.gate_history_conv_type == 'conv1x1_list':
            gate_history_conv_flops, history_conv_shape = count_conv2d_flops(conv2_out_shape, self.gate_hist_conv[0])
        elif self.args.gate_history_conv_type == 'conv1x1_res':
            gate_history_conv_flops, history_conv_shape = count_conv2d_flops(conv2_out_shape, self.gate_hist_conv)
            gate_history_conv_flops += count_relu_flops(history_conv_shape)[0]
        else:
            gate_history_conv_flops = 0
        return [conv1_flops, conv2_flops, downsample0_flops, gate_history_conv_flops], conv2_out_shape

    def forward(self, x, h_vec, h_map, t, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # TODO generate channel mask using local policy
        # TODO how to do history-fusion? Fused from history feature map? Or using LSTM here?

        adaptive_policy = not any([self.args.gate_all_zero_policy,
                                   self.args.gate_all_one_policy,
                                   self.args.gate_random_soft_policy,
                                   self.args.gate_random_hard_policy])

        mask = None
        h_vec_out = None
        h_map_out = None

        if self.args.gate_history and h_map is None:  # TODO t=0
            h_map = out

        if self.args.gate_history_detach:
            h_map = h_map.detach()

        if self.args.gate_history_conv_type in ['conv1x1','conv1x1bnrelu'] and "inline_test" not in kwargs:
            h_map = self.gate_hist_conv(h_map).detach()
            if self.args.gate_history_conv_type == 'conv1x1bnrelu':
                h_map = self.gate_hist_bnrelu(h_map)
        elif self.args.gate_history_conv_type == 'conv1x1_list' and "inline_test" not in kwargs:
            h_map = self.gate_hist_conv[t](h_map)
        elif self.args.gate_history_conv_type == 'conv1x1_res' and "inline_test" not in kwargs:
            h_map = (self.gate_hist_conv(h_map) + h_map)

        if self.args.gate_local_policy:
            factor = 3 if self.args.gate_history else 2
            if adaptive_policy:
                x_c = nn.AdaptiveAvgPool2d((1, 1))(x)
                x_c = torch.flatten(x_c, 1)

                if self.args.gate_history:
                    if h_vec is None:  # TODO t=0
                        h_vec = x_c
                    if self.args.fusion_type == "cat":
                        x_c = torch.cat([h_vec, x_c], dim=-1)
                    elif self.args.fusion_type == "add":
                        x_c = (x_c + h_vec) / 2
                    else:
                        exit("haven't implemented other vec fusion yet")
                    h_vec_out = x_c[:, -1*h_vec.shape[-1]:]

                x_c = self.gate_fc0(x_c)

                if self.args.gate_bn_between_fcs:
                    x_c = self.gate_bn(x_c)
                if self.args.gate_relu_between_fcs:
                    x_c = self.gate_relu(x_c)
                if self.args.gate_tanh_between_fcs:
                    x_c = self.tanh(x_c)

                x_c = self.gate_fc1(x_c)

                if self.args.gate_tanh:
                    x_c = self.tanh(x_c)
                    x_c = self.relu_not_inplace(x_c)
                elif self.args.gate_sigmoid:
                    x_c = self.sigmoid(x_c)

                use_hard = not self.args.gate_gumbel_use_soft

                if self.args.gate_history:
                    if self.args.gate_gumbel_softmax:
                        x_c2d = x_c.view(x.shape[0], self.num_channels, 3)
                        x_c2d = torch.log(F.softmax(x_c2d, dim=2))
                        mask2d = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=use_hard)
                        mask = mask2d  # TODO: B*C*3
                    elif self.args.gate_sem_hash:
                        x_c2d = x_c.view(x.shape[0], self.num_channels, 2)
                        is_training = "is_training" in kwargs and kwargs["is_training"]
                        curr_step = 0 if "curr_step" not in kwargs else kwargs["curr_step"]
                        # print(is_training)
                        mask2d = improved_sem_hash(x_c2d, is_training=is_training,
                                                   sem_hash_dense_random=self.args.gate_dense_random,
                                                   max_step=self.args.isemhash_max_step,
                                                   curr_step=curr_step)
                        h_mask = mask2d[:, :, 0]
                        c_mask = mask2d[:, :, 1]
                        mask = torch.stack([1+h_mask*c_mask-h_mask-c_mask, h_mask * (1-c_mask), c_mask], dim=-1)
                    elif self.args.gate_tanh:  # TODO using tanh
                        mask = x_c.view(x.shape[0], self.num_channels, 3)  # TODO: B*C*3
                    elif self.args.gate_sigmoid:
                        mask = x_c.view(x.shape[0], self.num_channels, 3)  # TODO: B*C*3
                else:
                    if self.args.gate_gumbel_sigmoid:
                        mask = gumbel_sigmoid(logits=x_c, tau=kwargs["tau"], hard=use_hard)
                        mask = torch.stack([1-mask, mask], dim=-1)
                    elif self.args.gate_gumbel_softmax:
                        x_c2d = x_c.view(x.shape[0], self.num_channels, 2)
                        x_c2d = torch.log(F.softmax(x_c2d, dim=2))
                        mask2d = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=use_hard)
                        mask = mask2d  # TODO: B*C*2
                    elif self.args.gate_tanh:  # TODO using tanh
                        mask = x_c.view(x.shape[0], self.num_channels, 2)  # TODO: B*C*2
                    elif self.args.gate_sigmoid:
                        mask = x_c.view(x.shape[0], self.num_channels, 2)  # TODO: B*C*2

            else:
                if self.args.gate_all_zero_policy:
                    mask = torch.zeros(x.shape[0], self.num_channels, factor, device=x.device)
                elif self.args.gate_all_one_policy:
                    mask = torch.ones(x.shape[0], self.num_channels, factor, device=x.device)
                elif self.args.gate_random_soft_policy:
                    mask = torch.rand(x.shape[0], self.num_channels, factor, device=x.device)
                elif self.args.gate_random_hard_policy:
                    # tmp_value = torch.rand(x.shape[0], self.num_channels, factor, device=x.device)
                    # mask = torch.zeros_like(tmp_value)
                    # mask[torch.where(tmp_value == torch.max(tmp_value, dim=-1, keepdim=True)[0])] = 1
                    tmp_value = torch.rand(x.shape[0], self.num_channels, device=x.device)
                    mask = torch.zeros(x.shape[0], self.num_channels, factor, device=x.device)

                    if len(self.args.gate_stoc_ratio)>0:
                        _ratio = self.args.gate_stoc_ratio
                    else:
                        _ratio = [0.333, 0.333, 0.334] if self.args.gate_history else [0.5, 0.5]
                    mask[:, :, 0][torch.where(tmp_value < _ratio[0])] = 1
                    if self.args.gate_history:
                        mask[:, :, 1][torch.where((tmp_value < _ratio[1]+_ratio[0]) & (tmp_value > _ratio[0]))] = 1
                        mask[:, :, 2][torch.where(tmp_value > _ratio[1]+_ratio[0])] = 1

            if self.args.gate_print_policy and "inline_test" in kwargs:
                print(mask[0, :max(1, self.num_channels // 64), -1])
                print(mask[-1, :max(1, self.num_channels // 64), -1])
                print()

            out = out * mask[:, :, -1].unsqueeze(-1).unsqueeze(-1)
            if self.args.gate_history:
                out = out + h_map * mask[:, :, -2].unsqueeze(-1).unsqueeze(-1)
                h_map_out = out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample0 is not None:
            y = self.downsample0(x)
            identity = self.downsample1(y)
        out += identity
        out = self.relu(out)

        return out, mask, h_vec_out, h_map_out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        if self.args.gate_local_policy:
            self.gate_fc0 = nn.Linear(width, self.args.gate_hidden_dim)
            self.gate_fc1 = nn.Linear(self.args.gate_hidden_dim, width)
            self.num_channels = width
            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)

    def forward(self, x, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        x_mid = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # TODO generate channel mask using local policy
        # TODO how to do history-fusion? Fused from history feature map? Or using LSTM here?
        if self.args.gate_local_policy:
            x_c = nn.AdaptiveAvgPool2d((1, 1))(x_mid)
            x_c = torch.flatten(x_c, 1)
            x_c = self.gate_fc0(x_c)
            x_c = self.gate_fc1(x_c)
            mask = gumbel_sigmoid(logits=x_c, tau=kwargs["tau"], hard=True)  # TODO: B*C
            out = out * mask.unsqueeze(-1).unsqueeze(-1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample0 is not None:
            y = self.downsample0(x)
            identity = self.downsample1(y)

        out += identity
        out = self.relu(out)

        return out


class GateNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, args=None):
        super(GateNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes_list_0, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1 = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list_0 * block.expansion:
            downsample0 = conv1x1(self.inplanes, planes_list_0 * block.expansion, stride)
            downsample1 = norm_layer(planes_list_0 * block.expansion)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list_0, stride, downsample0, downsample1, self.groups,
                            self.base_width, previous_dilation, norm_layer, args=self.args))
        self.inplanes = planes_list_0 * block.expansion
        for k in range(1, blocks):
            layers.append(block(self.inplanes, planes_list_0, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, args=self.args))
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

    def count_flops(self, input_data_shape, **kwargs):
        flops_list = []
        _B, _T, _C, _H, _W = input_data_shape
        input2d_shape = _B*_T, _C, _H, _W
        # for t in range(_T):
        #     flops, data_shape = count_conv2d_flops(input2d_shape, self.conv1)
        #     flops_sum += flops
        #
        #     flops ,data_shape = count_relu_flops()
        #
        #     idx=0
        #     for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
        #         for bi, block in enumerate(layers):
        #             flops, data_shape = block.count_flops(data_shape, masks[idx], **kwargs)
        #             flops_sum += flops
        #             idx += 1
        #
        #     flops_sum += count_fc_flops(input_data, self.fc)

        flops_conv1, data_shape = count_conv2d_flops(input2d_shape, self.conv1)
        data_shape = data_shape[0], data_shape[1], data_shape[2]//2, data_shape[3]//2 #TODO pooling
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                flops, data_shape = block.count_flops(data_shape, **kwargs)
                flops_list.append(flops)
        # print(list_sum(flops_list)+flops_conv1+512*200)
        # print(flops_list)
        return flops_list

    def forward(self, input_data, **kwargs):

        # TODO x.shape
        _B, _T, _C, _H, _W = input_data.shape
        out_list = []

        # #TODO history is block and conv-level state
        # history_rack = []
        # for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #     history_rack.append([])
        #     for _ in layers:
        #         history_rack[-1].append(None)


        if "tau" not in kwargs:
            print("tau not in kwargs. use default=1 (inline_test)")
            kwargs["tau"] = 1
            kwargs["inline_test"] = True

        mask_stack_list = []  # TODO list for t-dimension
        debug_stack_list = []
        h_vec_stack = []
        h_map_stack = []
        for _, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for _, block in enumerate(layers):
                mask_stack_list.append([])
                debug_stack_list.append([])
                h_vec_stack.append(None)
                h_map_stack.append(None)

        for t in range(_T):
            x = self.conv1(input_data[:, t])
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            idx = 0
            for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for bi, block in enumerate(layers):
                    x, mask, h_vec_stack[idx], h_map_stack[idx] = block(x, h_vec_stack[idx], h_map_stack[idx], t, **kwargs)
                    # history_rack[li][bi] = new_history
                    mask_stack_list[idx].append(mask)
                    if self.args.gate_debug:
                        debug_stack_list[idx].append([torch.norm(h_map_stack[idx],p=1)/h_map_stack[idx].numel(),
                                                      torch.max(torch.abs(h_map_stack[idx]))])
                    idx += 1

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)
            out_list.append(out)
        if self.args.gate_debug:
            print("norm")
            for t in range(_T):
                print("t:%d %s"%(t, " ".join(["%.3f"%debug_stack_list[block_i][t][0].detach().cpu().item() for block_i in range(len(debug_stack_list))])))
            print("max")
            for t in range(_T):
                print("t:%d %s"%(t, " ".join(["%.3f"%debug_stack_list[block_i][t][1].detach().cpu().item() for block_i in range(len(debug_stack_list))])))

            # if t == 0:
            #     mode_op += 1
        return torch.stack(out_list, dim=1), mask_stack_list


def _gatenet(arch, block, layers, pretrained, progress, **kwargs):
    model = GateNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        # TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
            # TODO layer4.0.downsample.X.weight -> layer4.0.downsampleX.weight
            if "downsample.0" in k:
                old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
            elif "downsample.1" in k:
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def gatenet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gatenet('gatenet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)


def gatenet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gatenet('gatenet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def gatenet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gatenet('gatenet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def gatenet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gatenet('gatenet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                    **kwargs)


def gatenet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gatenet('gatenet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                    **kwargs)
