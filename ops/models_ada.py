# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from ops.cnn3d.shufflenet3d import ShuffleNet3D
from ops.cnn3d.mobilenet3dv2 import MobileNet3DV2, mobilenet3d_v2
from ops.cnn3d.i3d_resnet import  I3D_ResNet, i3d_resnet

from tools.net_flops_table import feat_dim_dict

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

class TSN_Ada(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False,
                 rescale_to=224, rescale_pattern="L", args=None):
        super(TSN_Ada, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        #self.base_model_name = base_model

        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        # TODO(yue)
        self.rescale_to = rescale_to
        self.rescale_pattern = rescale_pattern
        self.args = args
        base_model = self.args.backbone_list[0] if len(self.args.backbone_list)>=1 else None
        self.base_model_name = base_model
        self.num_class = num_class

        #TODO (yue) for 3d-conv
        self.time_steps = self.num_segments // self.args.seg_len if self.args.cnn3d else self.num_segments

        # TODO(yue)
        if self.args.ada_reso_skip:

            shall_pretrain = not self.args.policy_from_scratch

            if self.args.cnn3d:
                if "shufflenet3d" in self.args.policy_backbone:
                    self.lite_backbone = ShuffleNet3D(groups=3, width_mult=float(self.args.policy_backbone.split("_")[1]))
                    if shall_pretrain:
                        #TODO(debug) Not sure how to do this, inflation?
                        a=1

                elif "mobilenet3dv2" in self.args.policy_backbone:
                    self.lite_backbone = mobilenet3d_v2(pretrained=shall_pretrain)
                elif "res3d" in self.args.policy_backbone:
                    self.lite_backbone = i3d_resnet(int(self.args.policy_backbone.split("res3d")[1]),pretrained=shall_pretrain)
            elif "efficientnet" in self.args.policy_backbone:
                if shall_pretrain:
                    self.lite_backbone = EfficientNet.from_pretrained(self.args.policy_backbone)
                else:
                    self.lite_backbone = EfficientNet.from_named(self.args.policy_backbone)
            else:
                self.lite_backbone = getattr(torchvision.models, self.args.policy_backbone)(shall_pretrain)
                #print(self.lite_backbone)

            self.lite_backbone.avgpool = nn.AdaptiveAvgPool2d(1)
            if "resnet" in self.args.policy_backbone:
                self.lite_backbone.last_layer_name = 'fc'
            elif "mobilenet_v2" in self.args.policy_backbone:
                self.lite_backbone.last_layer_name = 'classifier'
            elif "efficientnet" in self.args.policy_backbone:
                self.lite_backbone.last_layer_name = '_fc'
            elif self.args.cnn3d:
                if "shufflenet3d" in self.args.policy_backbone:
                    self.lite_backbone.last_layer_name = 'classifier'
                elif self.args.policy_backbone=="mobilenet3dv2":
                    self.lite_backbone.last_layer_name = 'classifier'
                elif "res3d" in self.args.policy_backbone:
                    self.lite_backbone.last_layer_name = 'fc'

            self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
            self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.args.hidden_dim, bias=True)

            self.multi_models = False
            if len(self.args.backbone_list) >= 1:
                self.multi_models = True
                self.base_model_list = nn.ModuleList() #[]
                self.new_fc_list = nn.ModuleList()#[]

            #self.reso_dim = len(self.args.backbone_list)
            self.reso_dim = 0
            for i in range(len(self.args.backbone_list)):
                self.reso_dim += self.args.ada_crop_list[i]

            if self.args.policy_also_backbone:
                self.reso_dim += 1
            self.skip_dim = len(self.args.skip_list)
            self.action_dim = self.reso_dim + self.skip_dim

            if self.args.no_skip:
                self.action_dim = self.reso_dim

        self._sanity_check(base_model, before_softmax, consensus_type, new_length, print_spec, modality)

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        self._do_if_flow_or_rgbdiff()

        self.consensus = ConsensusModule(consensus_type, args=self.args)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _sanity_check(self, base_model, before_softmax, consensus_type, new_length, print_spec, modality):
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
            img_feature_dim:    {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                           self.img_feature_dim)))

    def _do_if_flow_or_rgbdiff(self):
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        if self.args.ada_reso_skip:# and len(self.args.backbone_list) >= 1:
            for bbi, backbone_name in enumerate(self.args.backbone_list):
                shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[bbi].lower() != 'none'

                if 'efficientnet' in backbone_name:  # TODO(yue) pretrained
                    if shall_pretrain:
                        self.base_model_list.append(EfficientNet.from_pretrained(backbone_name))
                    else:  # TODO(yue) from scratch
                        self.base_model_list.append(EfficientNet.from_named(backbone_name))

                elif self.args.cnn3d:
                    if "shufflenet3d" in backbone_name:
                        self.base_model_list.append(ShuffleNet3D(3, width_mult=float(backbone_name.split("_")[1])))
                        if shall_pretrain:
                            #TODO(debug)
                            # self.base_model_list.append(EfficientNet.from_pretrained(backbone_name))
                            a=1
                    elif "mobilenet3dv2" in backbone_name:
                        self.base_model_list.append(mobilenet3d_v2(pretrained=shall_pretrain))
                    elif "res3d" in backbone_name:
                        self.base_model_list.append(i3d_resnet(depth=int(backbone_name.split("res3d")[1]), pretrained=shall_pretrain))
                else:
                    self.base_model_list.append(getattr(torchvision.models, backbone_name)(shall_pretrain))


                self.base_model_list[-1].avgpool = nn.AdaptiveAvgPool2d(1)
                if 'resnet' in backbone_name:
                    self.base_model_list[-1].last_layer_name = 'fc'
                elif backbone_name == 'mobilenet_v2':
                    self.base_model_list[-1].last_layer_name = 'classifier'
                elif 'efficientnet' in backbone_name:
                    self.base_model_list[-1].last_layer_name = '_fc'
                elif self.args.cnn3d:
                    if 'shufflenet3d' in backbone_name:
                        self.base_model_list[-1].last_layer_name = 'classifier'
                    elif "mobilenet3dv2" in backbone_name:
                        self.base_model_list[-1].last_layer_name = 'classifier'
                    elif "res3d" in backbone_name:
                        self.base_model_list[-1].last_layer_name = 'fc'
            return

        shall_pretrain = self.pretrain == 'imagenet'

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(shall_pretrain)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == "mobilenet_v2":
            #TODO(mobilenet)
            # from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            # self.base_model = mobilenet_v2(shall_pretrain)
            from archs.mobilenet_v2 import InvertedResidual
            self.base_model = getattr(torchvision.models, base_model)(shall_pretrain)

            self.base_model.last_layer_name = 'classifier'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif 'efficientnet' in base_model:
            if shall_pretrain:
                self.base_model = EfficientNet.from_pretrained(base_model)
            else:
                self.base_model = EfficientNet.from_named(base_model)
            self.base_model.last_layer_name = '_fc'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)


        elif self.args.cnn3d:
            if'shufflenet3d' in base_model:
                self.base_model = ShuffleNet3D(groups=3, width_mult=float(base_model.split("_")[1]))
                self.base_model.last_layer_name = 'classifier'
                self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
                self.input_size = 224
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]

            elif base_model == 'mobilenet3dv2':
                self.base_model = mobilenet3d_v2(pretrained = shall_pretrain)
                self.base_model.last_layer_name = 'classifier'
                self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
                self.input_size = 224
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]

            elif "res3d" in base_model:
                # TODO(yue) pretrained from ImageNet inflation
                self.base_model = i3d_resnet(depth=int(base_model.split("res3d")[1]),
                                             pretrained = shall_pretrain)
                self.base_model.last_layer_name = 'fc'
                self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
                self.input_size = 224
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):
        # TODO(yue)
        std = 0.001
        if self.args.ada_reso_skip:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.dropout))

            if self.args.frame_independent:
                self.linear = nn.Linear(self.policy_feat_dim, self.action_dim)
                self.lite_fc = nn.Linear(self.policy_feat_dim, num_class)
            else:
                self.linear = nn.Linear(self.args.hidden_dim, self.action_dim)
                self.lite_fc = nn.Linear(self.args.hidden_dim, num_class)
            if hasattr(self.linear, 'weight'):
                normal_(self.linear.weight, 0, std)
                constant_(self.linear.bias, 0)
            if hasattr(self.lite_fc, 'weight'):
                normal_(self.lite_fc.weight, 0, std)
                constant_(self.lite_fc.bias, 0)
        if self.multi_models:
            for j,base_model in enumerate(self.base_model_list):
                # feature_dim = getattr(base_model, base_model.last_layer_name).in_features

                if self.args.cnn3d and ("mobilenet3dv2"==self.args.backbone_list[j] or "shufflenet3d"in self.args.backbone_list[j]):
                    feature_dim = getattr(base_model, base_model.last_layer_name)[1].in_features
                elif "mobilenet_v2"==self.args.backbone_list[j]:
                    feature_dim = getattr(base_model, base_model.last_layer_name)[1].in_features
                else:
                    feature_dim = getattr(base_model, base_model.last_layer_name).in_features


                setattr(base_model, base_model.last_layer_name, nn.Dropout(p=self.dropout))
                new_fc = nn.Linear(feature_dim, num_class)
                if hasattr(new_fc, 'weight'):
                    normal_(new_fc.weight, 0, std)
                    constant_(new_fc.bias, 0)
                self.new_fc_list.append(new_fc)
            return None

        if self.base_model_name == None:
            return None

        if self.args.cnn3d and ("mobilenet3dv2"==self.base_model_name or "shufflenet3d"in self.base_model_name):
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
        elif "mobilenet_v2"==self.base_model_name:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        # feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Ada, self).train(mode)

        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")

            if self.args.ada_reso_skip:
                models=[self.lite_backbone]
                if self.multi_models:
                    models = models + self.base_model_list
            else:
                models=[self.base_model]
            for the_model in models:
                count = 0
                for m in the_model.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d): #TODO(yue)
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
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
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
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
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
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
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
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

    def forward(self, *argv, **kwargs):
        def backbone(input_data, the_base_model, new_fc):
            if self.args.cnn3d:
                # TODO(yue) input (B,T,C,H,W)
                # TODO(yue) view (B, T1, T2, C, H, W) -> (B*T1, T2, C, H, W) -> (B*T1, C, T2, H, W)   !(-1, C, H, W)
                # TODO(yue) base_out (B * T1, feat_dim)
                input_3d = input_data.view(input_data.shape[0], self.time_steps, self.args.seg_len,
                                      3, input_data.shape[-1], input_data.shape[-1])
                input_3d = input_3d.view(
                    (input_data.shape[0] * self.time_steps,) + input_3d.size()[2:])
                input_3d = input_3d.transpose(2, 1)
                base_out = the_base_model(input_3d)
            else:
                base_out = the_base_model(input_data.view((-1, 3 * self.new_length) + input_data.size()[-2:]))

            if new_fc is not None:
                base_out = new_fc(base_out)

            base_out = base_out.view((-1, self.time_steps) + base_out.size()[1:])

            return base_out

        batch_size = kwargs["input"][0].shape[0] # TODO(yue) input[0] B*(TC)*H*W

        if self.args.pyramid_boost:
            input_list = []
            _tmp_input = kwargs["input"][0]
            for i in range(len(self.base_model_list)):
                input_list.append(_tmp_input)
                _tmp_input = _tmp_input[:, :, :2, ::2]
        else:
            input_list = kwargs["input"]

        new_policy_input_offset = self.args.policy_input_offset
        for reso_i in range(self.args.policy_input_offset):
            if self.args.ada_crop_list[reso_i]>1:
                new_policy_input_offset -= 1
        #print(self.args.policy_input_offset, new_policy_input_offset)
        feat_lite = backbone(input_list[new_policy_input_offset], self.lite_backbone, None)
        hx = init_hidden(batch_size, self.args.hidden_dim)
        cx = init_hidden(batch_size, self.args.hidden_dim)
        logits_list = []
        lite_j_list = []

        base_out_list = []
        if self.multi_models:
            input_offset=0

            #print(input_list[0].shape)

            for bb_i,the_backbone in enumerate(self.base_model_list):
                if self.args.ada_crop_list[bb_i]==1:
                    base_out_list.append(backbone(input_list[input_offset], the_backbone, self.new_fc_list[bb_i]))
                    input_offset+=1
                else:
                    reso = self.args.reso_list[bb_i]
                    if self.args.ada_crop_list[bb_i]==5:
                        x_list=[0, 224-reso, 112-reso//2, 0, 224-reso]
                        y_list=[0, 0, 112-reso//2, 224-reso, 224-reso]
                    elif self.args.ada_crop_list[bb_i]==9:
                        x_list = [0, 112-reso//2, 224 - reso, 0, 112 - reso // 2, 224 - reso, 0, 112 - reso // 2, 224 - reso]
                        y_list = [0, 0, 0, 112-reso//2, 112-reso//2, 112-reso//2, 224 - reso, 224 - reso, 224 - reso]


                    for x,y in zip(x_list, y_list): #(B*T,C,H,W)
                        crop_input = input_list[0][:, :,  x:x+reso, y:y+reso] # crop
                        #print(reso, crop_input.shape)
                        base_out_list.append(backbone(crop_input, the_backbone, self.new_fc_list[bb_i]))

            # base_out_list = [backbone(input_list[i], self.base_model_list[i], self.new_fc_list[i]) for i in
            #                  range(len(self.base_model_list))]

        online_policy = False
        if not any([self.args.offline_lstm_all, self.args.offline_lstm_last, self.args.random_policy,
                    self.args.all_policy, self.args.real_scsampler]):
            online_policy = True
            r_list=[]

        remain_skip_vector = torch.zeros(batch_size,1)
        old_hx = None
        old_r_t = None

        if self.args.stabilize_order == False or online_policy == False:
            for t in range(self.time_steps):
                hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
                if self.args.frame_independent:
                    p_t = torch.log(F.softmax(self.linear(feat_lite[:, t]), dim=1))
                    j_t = self.lite_fc(feat_lite[:, t])
                else:
                    p_t = torch.log(F.softmax(self.linear(hx), dim=1))
                    j_t = self.lite_fc(hx)
                logits_list.append(p_t) #TODO as logit
                lite_j_list.append(j_t) #TODO as pred

                #TODO (yue) need a simple case to illustrate this
                if online_policy:
                    r_t_list=[]
                    for b_i in range(p_t.shape[0]):
                        r_t_item = F.gumbel_softmax(logits=p_t[b_i:b_i + 1], tau=kwargs["tau"], hard=True, eps=1e-10, dim=-1)
                        r_t_list.append(r_t_item)

                    r_t = torch.cat(r_t_list, dim=0)
                    #r_t = F.gumbel_softmax(logits=p_t, tau=kwargs["tau"], hard=True, eps=1e-10, dim=-1) #TODO(old version)
                    #TODO update states and r_t
                    if old_hx is not None:
                        take_bool = remain_skip_vector > 0.5
                        take_old = torch.tensor(take_bool, dtype=torch.float).cuda()
                        take_curr = torch.tensor(~take_bool, dtype=torch.float).cuda()
                        hx = old_hx * take_old + hx * take_curr
                        r_t = old_r_t * take_old + r_t * take_curr

                    # TODO update skipping_vector
                    for batch_i in range(batch_size):
                        for skip_i in range(self.action_dim-self.reso_dim):
                            #TODO(yue) first condition to avoid valuing skip vector forever
                            if remain_skip_vector[batch_i][0]<0.5 and r_t[batch_i][self.reso_dim + skip_i] > 0.5:
                                remain_skip_vector[batch_i][0] = self.args.skip_list[skip_i]
                    old_hx = hx
                    old_r_t = r_t
                    r_list.append(r_t)  # TODO as decision

                    remain_skip_vector = (remain_skip_vector - 1).clamp(0)

        else:
            for b_i in range(batch_size):
                hx = init_hidden(1, self.args.hidden_dim)
                cx = init_hidden(1, self.args.hidden_dim)
                old_hx = None
                old_r_t = None
                remain_skip_vector = torch.zeros(1)

                for t_i in range(self.time_steps):
                    hx, cx = self.rnn(feat_lite[b_i:b_i+1, t_i], (hx,cx))
                    p_t = torch.log(F.softmax(self.linear(hx), dim=1))
                    j_t = self.lite_fc(hx)
                    r_t_item = F.gumbel_softmax(logits=p_t, tau=kwargs["tau"], hard=True, eps=1e-10, dim=-1)

                    if old_hx is not None:
                        take_bool = remain_skip_vector[0] > 0.5
                        take_old = torch.tensor(take_bool, dtype=torch.float).cuda()
                        take_curr = torch.tensor(~take_bool, dtype=torch.float).cuda()
                        hx = old_hx * take_old + hx * take_curr
                        r_t_item = old_r_t * take_old + r_t_item * take_curr

                    for skip_i in range(self.action_dim-self.reso_dim):
                        if remain_skip_vector[0]<0.5 and r_t_item[0,self.reso_dim+skip_i]>0.5:
                            remain_skip_vector[0] = self.args.skip_list[skip_i]

                    old_hx = hx
                    old_r_t = r_t_item
                    r_list.append(r_t_item)  # TODO as decision
                    lite_j_list.append(j_t)  # TODO as pred
                    remain_skip_vector = (remain_skip_vector - 1).clamp(0)

            r_all = torch.cat(r_list, dim=0).reshape(batch_size, self.time_steps, self.action_dim)
            lite_j_list = torch.cat(lite_j_list, dim=0).reshape(batch_size, self.time_steps, self.num_class)

        if self.args.policy_also_backbone:
            if self.args.stabilize_order == False or online_policy == False:
                base_out_list.append(torch.stack(lite_j_list, dim=1))
            else:
                base_out_list.append(lite_j_list)

        if self.args.offline_lstm_last:  #TODO(yue) no policy - use policy net as backbone - just LSTM(last)
            if self.args.stabilize_order == False or online_policy == False:
                return lite_j_list[-1].squeeze(1), None
            else:
                return lite_j_list[:,-1], None
        elif self.args.offline_lstm_all: #TODO(yue) no policy - use policy net as backbone - just LSTM(average)
            if self.args.stabilize_order == False or online_policy == False:
                return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None
            else:
                return lite_j_list.mean(dim=1), None

        if self.args.real_scsampler:
            real_pred = base_out_list[0]
            lite_pred = torch.stack(lite_j_list, dim=1)
            output, ind = self.consensus(real_pred, lite_pred)
            return output.squeeze(1), ind, real_pred, lite_pred
        else:
            if self.args.random_policy: #TODO(yue) random policy
                r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                for i_bs in range(batch_size):
                    for i_t in range(self.time_steps):
                        r_all[i_bs, i_t, torch.randint(self.action_dim,[1])] = 1.0
            elif self.args.all_policy: #TODO(yue) all policy: take all
                r_all = torch.ones(batch_size, self.time_steps, self.action_dim).cuda()
            else: #TODO(yue) online policy
                # logits = torch.stack(logits_list, dim=1)  # N*T*K
                # r_all = F.gumbel_softmax(logits=logits, tau=kwargs["tau"], hard=True, eps=1e-10, dim=-1)
                if self.args.stabilize_order == False or online_policy == False:
                    r_all = torch.stack(r_list, dim=1)

            output = self.combine_logits(r_all, base_out_list)
            return output.squeeze(1), r_all

    def combine_logits(self, r, base_out_list):
        # TODO r           N, T, K  (0-origin, 1-low, 2-skip)
        # TODO base_out_list  <K * (N, T, C)
        offset = self.args.t_offset # 0
        batchsize = r.shape[0]
        total_num = torch.sum(r[:,offset:, 0:self.reso_dim], dim=[1,2]) + offset * batchsize

        res_base_list=[]
        for i,base_out in enumerate(base_out_list):
            r_like_base = r[:, :, i].unsqueeze(-1).expand_as(base_out) #TODO r[:,:,0] used to be, but wrong 01-30-2020
            if offset>0:
                if i==0:
                    r_like_base[:, :offset, :] = 1
                else:
                    r_like_base[:, :offset, :] = 0
            res_base = torch.sum(r_like_base * base_out, dim=1)
            res_base_list.append(res_base)
        if self.consensus_type!="scsampler":
            pred = torch.stack(res_base_list).sum(dim=0) / torch.clamp(total_num.unsqueeze(-1), 1)
        else:
            pred = self.consensus(torch.stack(res_base_list))
            pred = pred.squeeze(1)
        return pred

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

