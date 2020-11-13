"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from archs.efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)

from torch.nn.init import normal_, constant_
#TODO(yue)
from ops.utils import count_relu_flops, count_bn_flops, count_fc_flops, count_conv2d_flops

class PolicyBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer, shared, args):
        super(PolicyBlock, self).__init__()
        self.args = args
        self.norm_layer = norm_layer
        self.shared = shared
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        self.action_dim = out_factor

        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim * (args.gate_channel_ends - args.gate_channel_starts) // 64
        self.num_channels = out_dim // self.action_dim

        keyword = "%d_%d" % (in_dim, out_dim)
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim


        if self.args.shared_policy_net:
            self.gate_fc0 = self.shared[0][keyword]
        else:
            self.gate_fc0 = nn.Linear(in_dim, hidden_dim)
        if self.args.gate_bn_between_fcs:
            self.gate_bn = nn.BatchNorm1d(hidden_dim)
        if self.args.gate_relu_between_fcs:
            self.gate_relu = nn.ReLU(inplace=True)
        if self.args.shared_policy_net:
            self.gate_fc1 = self.shared[1][keyword]
        else:
            self.gate_fc1 = nn.Linear(hidden_dim, out_dim)

        # print("net0:", in_planes * in_factor)
        # print("net1:", out_planes * out_factor)
        # print("hidden", hidden_dim)
        # print("share0:", shared[0][keyword].weight.shape)
        # print("share1:", shared[1][keyword].weight.shape)

        if not self.args.shared_policy_net:
            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)



    def forward(self, x, **kwargs):
        # data preparation
        x_input = x

        if self.args.gate_reduce_type=="avg":
            x_c = nn.AdaptiveAvgPool2d((1, 1))(x_input)
        elif self.args.gate_reduce_type=="max":
            x_c = nn.AdaptiveMaxPool2d((1, 1))(x_input)
        x_c = torch.flatten(x_c, 1)
        _nt, _c = x_c.shape
        _t = self.args.num_segments
        _n = _nt // _t

        # history
        if self.args.gate_history:
            x_c_reshape = x_c.view(_n, _t, _c)
            h_vec = torch.zeros_like(x_c_reshape)
            h_vec[:, 1:] = x_c_reshape[:, :-1]
            h_vec = h_vec.view(_nt, _c)
            x_c = torch.cat([h_vec, x_c], dim=-1)

        # fully-connected embedding
        x_c = self.gate_fc0(x_c)
        if self.args.gate_bn_between_fcs:
            x_c = x_c.unsqueeze(-1)
            x_c = self.gate_bn(x_c)
            x_c = x_c.squeeze(-1)
        if self.args.gate_relu_between_fcs:
            x_c = self.gate_relu(x_c)
        x_c = self.gate_fc1(x_c)

        # gating operations
        x_c2d = x_c.view(x.shape[0], self.num_channels // self.args.granularity, self.action_dim)
        x_c2d = torch.log(F.softmax(x_c2d, dim=2).clamp(min=1e-8))
        # if 0 ==  x_c2d.get_device() and x_c2d.shape[1]==64:
        #     print(x_c2d[0, 1])
        mask = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=not self.args.gate_gumbel_use_soft)

        #TODO debug
        # if 0 == x_c2d.get_device() and x_c2d.shape[1] == 64:
        #     print(mask[0,1])
        #     print()

        if self.args.granularity>1:
            mask = mask.repeat(1, self.args.granularity, 1)
        if self.args.gate_channel_starts>0 or self.args.gate_channel_ends<64:
            full_channels = mask.shape[1] // (self.args.gate_channel_ends-self.args.gate_channel_starts) * 64
            channel_starts = full_channels // 64 * self.args.gate_channel_starts
            channel_ends = full_channels // 64 * self.args.gate_channel_ends
            outer_mask = torch.zeros(mask.shape[0], full_channels, mask.shape[2]).to(mask.device)
            outer_mask[:, :, -1] = 1.
            outer_mask[:, channel_starts:channel_ends] = mask

            return outer_mask
        else:
            return mask  # TODO: BT*C*ACT_DIM


def get_hmap(out, args, **kwargs):
    out_reshaped = out.view((-1, args.num_segments) + out.shape[1:])

    if args.gate_history:
        h_map_reshaped = torch.zeros_like(out_reshaped)  # TODO(yue) n, t, c, h, w
        h_map_reshaped[:, 1:] = out_reshaped[:, :-1]
    else:
        return None

    if args.gate_history_detach:
        h_map_reshaped = h_map_reshaped.detach()

    h_map_updated = h_map_reshaped.view((-1,) + out_reshaped.shape[2:])
    return h_map_updated


def fuse_out_with_mask(out, mask, h_map, args):
    out = out * mask[:, :, -1].unsqueeze(-1).unsqueeze(-1)
    if args.gate_history:
        out = out + h_map * mask[:, :, -2].unsqueeze(-1).unsqueeze(-1)
    return out







class AdaMBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """AdaMBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        mask=None #TODO(yue)

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x, mask

    #TODO(yue)
    def count_flops(self, input_data_shape):
        if self._block_args.expand_ratio != 1:
            conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self._expand_conv, is_effnet_conv=True)
        else:
            conv1_flops = 0
            conv1_out_shape = input_data_shape
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self._depthwise_conv, is_effnet_conv=True)
        conv3_flops, conv3_out_shape = count_conv2d_flops(conv2_out_shape, self._project_conv, is_effnet_conv=True)
        return [conv1_flops, conv2_flops, conv3_flops], conv3_out_shape

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    Example:


        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None, args=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.args=args #TODO(yue)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(AdaMBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(AdaMBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x


    def count_flops(self, input_data_shape):
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        """
        flops_list=[]
        shape_list=[]
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self._conv_stem, is_effnet_conv=True)
        flops_list.append([conv1_flops])
        shape_list.append(conv1_out_shape)
        conv_i_out_shape = conv1_out_shape
        for idx, block in enumerate(self._blocks):
            conv_i_flops, conv_i_out_shape = block.count_flops(conv_i_out_shape)
            flops_list.append(conv_i_flops)
            shape_list.append(conv_i_out_shape)
        conv3_flops, conv3_out_shape = count_conv2d_flops(conv_i_out_shape, self._conv_head, is_effnet_conv=True)
        flops_list.append([conv3_flops])
        shape_list.append(conv3_out_shape)
        return flops_list, shape_list


    @classmethod
    def from_name(cls, model_name, args, in_channels=3, **override_params):
        """create an efficientnet model according to name.
        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params, args)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, args, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.
        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, args, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000),
                                advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)