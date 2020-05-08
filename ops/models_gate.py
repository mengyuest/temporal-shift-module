from torch import nn
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import ops.gatenet as gatenet

from tools.net_flops_table import feat_dim_dict


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

class TSN_Gate(nn.Module):
    def __init__(self, args):
        super(TSN_Gate, self).__init__()
        self.num_segments = args.num_segments

        self.args = args
        self.base_model_name = self.args.backbone_list[0]
        self.num_class = args.num_class

        self.reso_dim = 1
        self.skip_dim = 0
        self.action_dim = 1  # TODO(yue)

        if not (self.args.gate and self.args.gate_local_policy):
            self._prepare_policy_net()
        self.base_model_list = nn.ModuleList()
        self.new_fc_list = nn.ModuleList()

        self._prepare_base_model()
        self._prepare_fc(args.num_class)

        self._enable_pbn = False

    def _prep_a_net(self, model_name, shall_pretrain):
        if "gatenet" in model_name:
            model = getattr(gatenet, model_name)(shall_pretrain, args=self.args)
            model.last_layer_name = 'fc'
        elif "efficientnet" in model_name:
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
            model.last_layer_name = "_fc"
        else:
            exit("I don't how to prep this net; see models_gate.py:: _prep_a_net")
        return model

    def _prepare_policy_net(self):
        shall_pretrain = not self.args.policy_from_scratch
        self.lite_backbone = self._prep_a_net(self.args.policy_backbone, shall_pretrain)
        self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
        self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.args.hidden_dim, bias=True)

    def _prepare_base_model(self):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
        model = self._prep_a_net(self.args.backbone_list[0], shall_pretrain)
        self.base_model_list.append(model)


    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model

        i_do_need_a_policy_network = not (self.args.gate and self.args.gate_local_policy)

        if i_do_need_a_policy_network:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.args.dropout))
            feed_dim = self.args.hidden_dim if not self.args.frame_independent else self.policy_feat_dim
            self.linear = make_a_linear(feed_dim, self.action_dim)
            self.lite_fc = make_a_linear(feed_dim, num_class)

        feature_dim = getattr(self.base_model_list[0], self.base_model_list[0].last_layer_name).in_features
        new_fc = make_a_linear(feature_dim, num_class)
        self.new_fc_list.append(new_fc)
        setattr(self.base_model_list[0], self.base_model_list[0].last_layer_name, nn.Dropout(p=self.args.dropout))

    def forward(self, *argv, **kwargs):
        input_data = kwargs["input"][0]  # TODO(yue) B * (TC) * H * W
        _b, _tc, _h, _w = input_data.shape
        _t, _c = _tc // 3, 3

        if "tau" not in kwargs:
            kwargs["tau"] = None

        feat, mask_stack_list = self.base_model_list[0](input_data.view(_b, _t, _c, _h, _w), tau=kwargs["tau"])
        base_out = self.new_fc_list[0](feat.view(_b * _t, -1)).view(_b, _t, -1)

        output = base_out.mean(dim=1).squeeze(1)

        for i in range(len(mask_stack_list)):
            mask_stack_list[i] = torch.stack(mask_stack_list[i], dim=1)

        return output, mask_stack_list, None, torch.stack([base_out], dim=1)


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Gate, self).train(mode)
        if mode and len(self.args.frozen_layers) > 0 and self.args.freeze_corr_bn:
            models = list(self.base_model_list)

            for the_model in models:
                for layer_idx in self.args.frozen_layers:
                    for km in the_model.named_modules():
                        k, m = km
                        if layer_idx == 0:
                            if "bn1" in k and "layer" not in k and (
                                    isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):  # TODO(yue)
                                m.eval()
                                m.weight.requires_grad = False
                                m.bias.requires_grad = False
                        else:
                            if "layer%d" % (layer_idx) in k and (
                                    isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):  # TODO(yue)
                                m.eval()
                                m.weight.requires_grad = False
                                m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]