from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from ops.cnn3d.mobilenet3dv2 import MobileNet3DV2, mobilenet3d_v2
from ops.cnn3d.i3d_resnet import  I3D_ResNet, i3d_resnet
import ops.dmynet as dmynet
from ops.dmynet import Conv2dMY
import ops.dhsnet as dhsnet
from ops.dhsnet import Conv2dHS
import ops.msdnet as msdnet
import ops.mernet as mernet
import ops.csnet as csnet

from tools.net_flops_table import feat_dim_dict

from os.path import join as ospj
import common


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

class TSN_Ada(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', consensus_type='avg', before_softmax=True, dropout=0.8,
                 crop_num=1, partial_bn=True, pretrain='imagenet', is_shift=False, shift_div=8, shift_place='blockres',
                 fc_lr5=False, temporal_pool=False, non_local=False, args=None):
        super(TSN_Ada, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        # TODO(yue)
        self.args = args
        self.rescale_to=args.rescale_to
        if self.args.ada_reso_skip:
            base_model = self.args.backbone_list[0] if len(self.args.backbone_list)>=1 else None
        self.base_model_name = base_model
        self.num_class = num_class
        self.multi_models=False

        #TODO (yue) for 3d-conv
        self.time_steps = self.num_segments // self.args.seg_len if self.args.cnn3d else self.num_segments

        if self.args.ada_reso_skip:
            self._prepare_policy_net()
            self._extends_to_multi_models()

        self._prepare_base_model(base_model)
        self._prepare_fc(num_class)

        self.consensus = ConsensusModule(consensus_type, args=self.args)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _extends_to_multi_models(self):
        if len(self.args.backbone_list) >= 1:
            self.multi_models = True
            self.base_model_list = nn.ModuleList()
            self.new_fc_list = nn.ModuleList()

    def _prep_a_net(self, model_name, shall_pretrain):
        if self.args.dmy and "dmynet" in model_name:
            model = getattr(dmynet, model_name)(shall_pretrain, num_filters_list = self.args.num_filters_list,
                                                last_conv_same = self.args.last_conv_same)
            model.last_layer_name = 'fcs'
        elif self.args.dhs and "dhsnet" in model_name:
            model = getattr(dhsnet, model_name)(shall_pretrain, num_filters_list=self.args.num_filters_list,
                                                args=self.args)
            model.last_layer_name = 'fc'
        elif self.args.msd and "msdnet" in model_name:
            pretrained_path = ospj(common.PYTORCH_CKPT_DIR, "msdnet-step4-block5.pth") if shall_pretrain else None
            model = getattr(msdnet, "create_msdnet")(pretrained_path=pretrained_path,
                                                     gradient_equilibrium=self.args.gradient_equilibrium)
        elif self.args.mer and "mernet" in model_name:
            model = getattr(mernet, model_name)(shall_pretrain)
        elif self.args.csn and "csn" in model_name:
            model = getattr(csnet, model_name)(shall_pretrain)
            model.last_layer_name = 'fc'
        elif "mobilenet3dv2" in model_name:
            model = mobilenet3d_v2(shall_pretrain)
            model.last_layer_name = 'classifier'
        elif "res3d" in model_name:
            model = i3d_resnet(int(model_name.split("res3d")[1]), shall_pretrain)
            model.last_layer_name = 'fc'
        elif "efficientnet" in model_name:
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
            model.last_layer_name = "_fc"
        else:
            model = getattr(torchvision.models, model_name)(shall_pretrain)
            if "resnet" in model_name:
                model.last_layer_name = 'fc'
            elif "mobilenet_v2" in model_name:
                model.last_layer_name = 'classifier'
        return model

    def _get_resolution_dimension(self):
        reso_dim = 0
        if self.args.dmy:
            reso_dim = len(self.args.num_filters_list)
        elif self.args.dhs:
            reso_dim = len(self.args.num_filters_list)
        elif self.args.msd:
            reso_dim = len(self.args.msd_indices_list) #TODO 0,1,2; 0,2,3...
        elif self.args.mer:
            reso_dim = len(self.args.mer_indices_list) #TODO 1,2,3 for exit-1, exit-2 and last one
        elif self.args.csn:
            reso_dim = len(self.args.backbone_list)  # TODO
        else:
            for i in range(len(self.args.backbone_list)):
                reso_dim += self.args.ada_crop_list[i]
        if self.args.policy_also_backbone:
            reso_dim += 1
        return reso_dim

    def _make_a_shift(self, base_model):
        if "resnet" in base_model:
            print('Adding temporal shift...')
            from ops.temporal_shift import make_temporal_shift
            make_temporal_shift(self.base_model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
        else:
            from ops.temporal_shift import TemporalShift
            from archs.mobilenet_v2 import InvertedResidual
            for m in self.base_model.modules():
                if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                    if self.print_spec:
                        print('Adding temporal shift... {}'.format(m.use_res_connect))
                    m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)

    def _prepare_policy_net(self):
        shall_pretrain = not self.args.policy_from_scratch
        self.lite_backbone = self._prep_a_net(self.args.policy_backbone, shall_pretrain)
        self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
        self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.args.hidden_dim, bias=True)

        self.reso_dim = self._get_resolution_dimension()
        self.skip_dim = len(self.args.skip_list)
        self.action_dim = self.reso_dim + self.skip_dim

    def _prepare_base_model(self, base_model):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if self.args.ada_reso_skip:
            shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
            if self.args.separate_dmy:
                for _ in self.args.num_filters_list:
                    model = self._prep_a_net(self.args.backbone_list[0], shall_pretrain)
                    self.base_model_list.append(model)
            elif any([self.args.dmy, self.args.dhs, self.args.msd, self.args.mer]):
                model = self._prep_a_net(self.args.backbone_list[0], shall_pretrain)
                self.base_model_list.append(model)
            else:
                for bbi, backbone_name in enumerate(self.args.backbone_list):
                    model = self._prep_a_net(backbone_name, shall_pretrain)
                    self.base_model_list.append(model)
        else:
            self.base_model = self._prep_a_net(base_model, self.pretrain == 'imagenet')
            if self.is_shift:
                self._make_a_shift(base_model)

    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model

        if self.args.ada_reso_skip:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.dropout))
            feed_dim = self.args.hidden_dim if not self.args.frame_independent else self.policy_feat_dim
            self.linear = make_a_linear(feed_dim, self.action_dim)
            self.lite_fc = make_a_linear(feed_dim, num_class)

        if self.multi_models:
            if self.args.dmy:
                for bbi, base_model in enumerate(self.base_model_list):
                    for fc_i in range(len(self.args.num_filters_list)):
                        feature_dim = getattr(base_model, base_model.last_layer_name)[fc_i].in_features
                        new_fc = make_a_linear(feature_dim, num_class)

                        if fc_i % len(self.base_model_list) == bbi % len(self.base_model_list):
                            self.new_fc_list.append(new_fc)
                        if self.args.last_conv_same:
                            break
                    setattr(base_model, base_model.last_layer_name, torch.nn.ModuleList([nn.Dropout(p=self.dropout) for _ in self.args.num_filters_list]))

            elif self.args.dhs:
                feature_dim = getattr(self.base_model_list[0], self.base_model_list[0].last_layer_name).in_features
                new_fc = make_a_linear(feature_dim, num_class)
                self.new_fc_list.append(new_fc)
                setattr(self.base_model_list[0], self.base_model_list[0].last_layer_name, nn.Dropout(p=self.dropout))
            else:
                multi_fc_list = [None]
                if self.args.msd:
                    multi_fc_list = self.args.msd_indices_list
                elif self.args.mer:
                    multi_fc_list = self.args.mer_indices_list

                for bbi, base_model in enumerate(self.base_model_list):
                    for fc_i, exit_index in enumerate(multi_fc_list):
                        if self.args.msd:
                            feature_dim = getattr(base_model, "classifier")[exit_index].linear.in_features
                        elif self.args.mer:
                            last_layer_name = "fc%d" % exit_index
                            feature_dim = getattr(base_model, last_layer_name).in_features
                        else:
                            last_layer_name = base_model.last_layer_name
                            if self.args.backbone_list[bbi] in ["mobilenet3dv2", "mobilenet_v2"]:
                                feature_dim = getattr(base_model, last_layer_name)[1].in_features
                            else:
                                feature_dim = getattr(base_model, last_layer_name).in_features

                        new_fc = make_a_linear(feature_dim, num_class)
                        self.new_fc_list.append(new_fc)
                        if self.args.msd:
                            base_model.classifier[self.args.msd_indices_list[fc_i]].linear = nn.Dropout(p=self.dropout)
                        else:  # TODO mer and normal
                            setattr(base_model, last_layer_name, nn.Dropout(p=self.dropout))

        elif self.base_model_name is not None:
            if self.args.dmy:
                the_fc_i = self.args.default_signal if not self.args.last_conv_same else 0
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[the_fc_i].in_features
                setattr(self.base_model, self.base_model.last_layer_name,
                        torch.nn.ModuleList([nn.Dropout(p=self.dropout) for _ in self.args.num_filters_list]))
            else:
                if self.args.cnn3d and "mobilenet3dv2"==self.base_model_name:
                    feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
                elif "mobilenet_v2"==self.base_model_name:
                    feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
                else:
                    feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features

                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = make_a_linear(feature_dim, num_class)

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
                bn_scale = len(self.args.num_filters_list)
                for m in the_model.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d): #TODO(yue)
                        count += 1
                        if count >= (2*bn_scale if self._enable_pbn else bn_scale):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
        if mode and len(self.args.frozen_layers) > 0 and self.args.freeze_corr_bn:
            if self.args.ada_reso_skip:
                models = []
                if self.multi_models:
                    models = models + list(self.base_model_list)
            else:
                models = [self.base_model]
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
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d)\
                    or isinstance(m, Conv2dMY) or isinstance(m, Conv2dHS):
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

    def backbone(self, input_data, the_base_model, new_fc, signal=-1, indices_list=[], boost=False, b_t_c=False):
        if boost:
            _bt, _, _h, _w = input_data.shape
        else:
            _b, _tc, _h, _w = input_data.shape  # TODO(yue) input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
        if self.args.cnn3d:
            input_3d = input_data.view(_b, self.time_steps, self.args.seg_len, _c, _h, _w)  # TODO(yue) T=T1*T2
            input_3d = input_3d.view(_b * self.time_steps, self.args.seg_len, _c, _h, _w)  # TODO(yue) B,T1 -> B*T1
            input_3d = input_3d.transpose(2, 1)  # TODO(yue) view  (B*T1, C, T2, H, W)
            feat = the_base_model(input_3d)  # TODO(yue) feat  (B*T1, feat_dim)
        else:
            if boost:
                input_2d = input_data
            elif b_t_c:
                input_b_t_c = input_data.view(_b, _t, _c, _h, _w)
            else:
                input_2d = input_data.view(_b * _t, _c, _h, _w)

            if b_t_c:
                feat = the_base_model(input_b_t_c, signal=signal)
            elif any([self.args.dmy, self.args.msd, self.args.mer]) and \
                    ((self.args.uno_reso and len(indices_list) > 0) or (signal is not None and signal >= 0)):
                feat = the_base_model(input_2d, signal=signal)
            else:
                feat = the_base_model(input_2d)

        if (self.args.msd or self.args.mer) and self.args.uno_reso and len(indices_list)>0:
            _feat_out_list = [f.view(_b, _t, -1) for f in feat]
            _base_out_list = [new_fc[i](feat[ind_i]).view(_b, _t, -1) for i, ind_i in enumerate(indices_list)]
            return _feat_out_list, _base_out_list
        else:
            _base_out = None
            if boost:
                if new_fc is not None:
                    _base_out = new_fc(feat)
            elif b_t_c:
                if new_fc is not None:
                    _base_out = new_fc(feat.view(_b * _t, -1)).view(_b, _t, -1)
            else:
                if new_fc is not None:
                    _base_out = new_fc(feat).view(_b, _t, -1)
                feat = feat.view(_b, _t, -1)
            return feat, _base_out

    def get_lite_j_and_r(self, input_list, online_policy, tau):

        feat_lite, _ = self.backbone(input_list[self.args.policy_input_offset], self.lite_backbone, None)

        r_list = []
        lite_j_list = []
        logits_list = []
        batch_size = feat_lite.shape[0]
        hx = init_hidden(batch_size, self.args.hidden_dim)
        cx = init_hidden(batch_size, self.args.hidden_dim)

        remain_skip_vector = torch.zeros(batch_size, 1)
        old_hx = None
        old_r_t = None

        for t in range(self.time_steps):
            if self.args.frame_independent:
                feat_t = feat_lite[:, t]
            else:
                hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
                feat_t = hx
            p_t = torch.log(F.softmax(self.linear(feat_t), dim=1))
            j_t = self.lite_fc(feat_t)
            logits_list.append(p_t)  # TODO as logit
            lite_j_list.append(j_t)  # TODO as pred

            # TODO (yue) need a simple case to illustrate this
            if online_policy:
                r_t = torch.cat(
                    [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                # r_t = F.gumbel_softmax(logits=p_t, tau=kwargs["tau"], hard=True)

                # TODO update states and r_t
                if old_hx is not None:
                    take_bool = remain_skip_vector > 0.5
                    take_old = torch.tensor(take_bool, dtype=torch.float).cuda()
                    take_curr = torch.tensor(~take_bool, dtype=torch.float).cuda()
                    hx = old_hx * take_old + hx * take_curr
                    r_t = old_r_t * take_old + r_t * take_curr

                # TODO update skipping_vector
                for batch_i in range(batch_size):
                    for skip_i in range(self.action_dim - self.reso_dim):
                        # TODO(yue) first condition to avoid valuing skip vector forever
                        if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][self.reso_dim + skip_i] > 0.5:
                            remain_skip_vector[batch_i][0] = self.args.skip_list[skip_i]
                old_hx = hx
                old_r_t = r_t
                r_list.append(r_t)  # TODO as decision
                remain_skip_vector = (remain_skip_vector - 1).clamp(0)
        if online_policy:
            return lite_j_list, torch.stack(r_list, dim=1)
        else:
            return lite_j_list, None

    def using_online_policy(self):
        if any([self.args.offline_lstm_all, self.args.offline_lstm_last]):
            return False
        elif any([self.args.random_policy, self.args.all_policy, self.args.real_all_policy, self.args.distill_policy]):
            return False
        elif self.args.real_scsampler:
            return False
        else:
            return True

    def get_feat_and_pred(self, input_list, r_all):
        feat_out_list = []
        base_out_list = []
        ind_list = []
        if self.args.dmy:
            for fc_i in range(len(self.args.num_filters_list)):
                if self.args.boost:
                    _b, _tc, _h, _w = input_list[fc_i].shape
                    indices = (r_all[:, :, fc_i] == 1).nonzero()
                    if indices.shape[0] == 0:
                        feat_out_list.append(None)
                        base_out_list.append(None)
                        ind_list.append(None)
                        continue
                    input_data = (input_list[fc_i].view(_b, _tc//3, 3, _h, _w))[indices[:, 0], indices[:, 1]]
                    input_data = input_data.view(-1, 3, _h, _w)
                    ind_list.append(torch.tensor([x[0] * _tc // 3 + x[1] for x in indices]))
                else:
                    input_data = input_list[fc_i]
                last_layer_fc_i = fc_i if not self.args.last_conv_same else 0
                feat_out, base_out = self.backbone(input_data, self.base_model_list[fc_i % len(self.base_model_list)],
                                                self.new_fc_list[last_layer_fc_i], signal=fc_i, boost=self.args.boost)
                feat_out_list.append(feat_out)
                base_out_list.append(base_out)
        elif self.args.dhs:
            input_data = input_list[0]
            feat_out, base_out = self.backbone(input_data, self.base_model_list[0],
                                               self.new_fc_list[0], signal=r_all, boost=self.args.boost, b_t_c=True)
            feat_out_list.append(feat_out)
            base_out_list.append(base_out)

        elif any([self.args.msd, self.args.mer]):
            iter_list = self.args.msd_indices_list if self.args.msd else self.args.mer_indices_list
            if self.args.uno_reso:
                feat_out_list, base_out_list = self.backbone(input_list[0], self.base_model_list[0],
                                                             self.new_fc_list, signal=None, indices_list=iter_list)
            else:
                for fc_i in range(len(iter_list)):
                    feat_out, base_out = self.backbone(input_list[fc_i], self.base_model_list[0],
                                                       self.new_fc_list[fc_i], signal=iter_list[fc_i])
                    feat_out_list.append(feat_out)
                    base_out_list.append(base_out)
        else:
            for bb_i, the_backbone in enumerate(self.base_model_list):
                feat_out, base_out = self.backbone(input_list[bb_i], the_backbone, self.new_fc_list[bb_i])
                feat_out_list.append(feat_out)
                base_out_list.append(base_out)
        return feat_out_list, base_out_list, ind_list

    def forward(self, *argv, **kwargs):
        if not self.args.ada_reso_skip:  # TODO simple TSN
            _, base_out = self.backbone(kwargs["input"][0], self.base_model, self.new_fc, signal=self.args.default_signal)
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

        input_list = kwargs["input"]
        batch_size = input_list[0].shape[0]  # TODO(yue) input[0] B*(TC)*H*W
        lite_j_list, r_all = self.get_lite_j_and_r(input_list, self.using_online_policy(), kwargs["tau"])

        if self.multi_models:
            if self.args.dhs and self.args.random_policy:
                r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                for i_bs in range(batch_size):
                    for i_t in range(self.time_steps):
                        r_all[i_bs, i_t, torch.randint(self.action_dim, [1])] = 1.0
            feat_out_list, base_out_list, ind_list = self.get_feat_and_pred(input_list, r_all)
        else:
            feat_out_list, base_out_list, ind_list  = [], [], []

        if self.args.policy_also_backbone:
            base_out_list.append(torch.stack(lite_j_list, dim=1))

        if self.args.offline_lstm_last:  #TODO(yue) no policy - use policy net as backbone - just LSTM(last)
            return lite_j_list[-1].squeeze(1), None, None, None

        elif self.args.offline_lstm_all: #TODO(yue) no policy - use policy net as backbone - just LSTM(average)
            return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None, None, None

        elif self.args.real_scsampler:
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
            elif self.args.all_policy or self.args.real_all_policy: #TODO(yue) all policy: take all
                r_all = torch.ones(batch_size, self.time_steps, self.action_dim).cuda()
            elif self.args.distill_policy: #TODO(yue) all policy: take all
                r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                r_all[:, :, 0] = 1.0

            output = self.combine_logits(r_all, base_out_list, ind_list)
            if self.args.save_meta and self.args.save_all_preds:
                return output.squeeze(1), r_all, torch.stack(base_out_list,dim=1)
            else:
                if self.args.last_conv_same:
                    return output.squeeze(1), r_all, torch.stack(feat_out_list,dim=1), torch.stack(base_out_list,dim=1)
                else:
                    if self.args.boost:
                        return output.squeeze(1), r_all, None, None
                    else:
                        return output.squeeze(1), r_all, None, torch.stack(base_out_list, dim=1)

    def combine_logits(self, r, base_out_list, ind_list):
        # TODO r                N, T, K
        # TODO base_out_list  < K * (N, T, C)
        if self.args.boost:
            if len(ind_list)==0 or all([x is None for x in ind_list]):
                return torch.zeros(r.shape[0], self.num_class).cuda()
            pred_tensor2d = torch.cat([x for x in base_out_list if x is not None], dim=0)  # TODO N'T'*K
            ind_tensor2d = torch.cat([x.unsqueeze(-1) for x in ind_list if x is not None], dim=0)
            ind_tensor1d = ind_tensor2d.squeeze(-1)

            if ind_tensor1d.shape[0] == 0:  # TODO(yue) this batch is fei le => return zeros
                return torch.zeros(r.shape[0], self.num_class).cuda()
            pred_tensor = torch.zeros(r.shape[0], r.shape[1], self.num_class).cuda()
            pred_tensor[ind_tensor1d // r.shape[1], ind_tensor1d % r.shape[1]] = pred_tensor2d
            t_tensor = torch.sum(r[:, :, :self.reso_dim], dim=[1, 2]).unsqueeze(-1).clamp(1)
            return pred_tensor.sum(dim=1) / t_tensor
        elif self.args.dhs:
            return base_out_list[0].mean(dim=1)
        else:
            pred_tensor = torch.stack(base_out_list, dim=2)
            r_tensor = r[:, :, :self.reso_dim].unsqueeze(-1)
            t_tensor = torch.sum(r[:, :, :self.reso_dim], dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
            return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor

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