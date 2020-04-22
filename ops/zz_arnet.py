import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torchvision
from efficientnet_pytorch import EfficientNet


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

feat_dim_dict = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "mobilenet_v2": 1280,
    "efficientnet-b0": 1280,
    "efficientnet-b1": 1280,
    "efficientnet-b2": 1408,
    "efficientnet-b3": 1536,
    "efficientnet-b4": 1792,
    "efficientnet-b5": 2048,
    }

class SegmentConsensus(torch.autograd.Function):

    def __init__(self):
        self.dim = 1
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        output = input_tensor.mean(dim=self.dim, keepdim=True)
        return output

    def backward(self, grad_output):
        grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self):
        super(ConsensusModule, self).__init__()

    def forward(self, input, lite_input=None):
        return SegmentConsensus()(input)


class ARNet(nn.Module):
    def __init__(self, num_class, num_segments, base_model, pretrain='imagenet', args=None):
        super(ARNet, self).__init__()
        self.num_segments = num_segments
        self.fc_lr5 = True
        self.reshape = True
        self.before_softmax = True
        self.dropout = 0.5
        self.pretrain = pretrain
        self.hidden_dim = 512
        self.args = args  # TODO(yue)

        if self.args.ada_reso_skip:
            base_model = self.args.backbone_list[0] if len(self.args.backbone_list)>=1 else None
        self.base_model_name = base_model
        self.num_class = num_class
        self.multi_models=False


        if self.args.ada_reso_skip:
            self._prepare_policy_net()

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        self.consensus = ConsensusModule()

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _get_component(self, model_name, shall_pretrain):
        if "efficientnet" in model_name:
            last_layer_name = '_fc'
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
        else:
            if "mobilenet_v2" in model_name:
                last_layer_name = 'classifier'
            elif 'resnet' in model_name:
                last_layer_name = 'fc'
            else:
                raise ValueError('Unknown base model: {}'.format(model_name))
            model = getattr(torchvision.models, model_name)(shall_pretrain)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.last_layer_name = last_layer_name
        return model

    def _prepare_policy_net(self):
        self.lite_backbone = self._get_component(self.args.policy_backbone, not self.args.policy_from_scratch)

        self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
        self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.hidden_dim, bias=True)

        if len(self.args.backbone_list) >= 1:
            self.multi_models = True
            self.base_model_list = nn.ModuleList()
            self.new_fc_list = nn.ModuleList()

        self.reso_dim = len(self.args.backbone_list)
        if self.args.policy_also_backbone:
            self.reso_dim += 1
        self.skip_dim = len(self.args.skip_list)
        self.action_dim = self.reso_dim + self.skip_dim

    def _prepare_base_model(self, base_model):

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        if self.args.ada_reso_skip:
            for bbi, backbone_name in enumerate(self.args.backbone_list):
                shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[bbi].lower() != 'none'
                backbone_model = self._get_component(backbone_name, shall_pretrain)
                self.base_model_list.append(backbone_model)
        else:
            self.base_model = self._get_component(base_model, self.pretrain == 'imagenet')

    def _initialized_nn_linear(self, feature_dim, action_dim):
        linear_classifier = nn.Linear(feature_dim, action_dim)
        normal_(linear_classifier.weight, 0, 0.001)
        constant_(linear_classifier.bias, 0)
        return linear_classifier

    def _prep_backbone_and_new_fc(self, backbone, backbone_name, num_class):
        if "mobilenet_v2" == backbone_name:
            feature_dim = getattr(backbone, backbone.last_layer_name)[1].in_features
        else:
            feature_dim = getattr(backbone, backbone.last_layer_name).in_features
        setattr(backbone, backbone.last_layer_name, nn.Dropout(p=self.dropout))
        new_fc = self._initialized_nn_linear(feature_dim, num_class)
        return new_fc

    def _prepare_tsn(self, num_class):
        if self.args.ada_reso_skip:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.dropout))
            self.linear = self._initialized_nn_linear(self.hidden_dim, self.action_dim)
            self.lite_fc = self._initialized_nn_linear(self.hidden_dim, num_class)

        if self.multi_models:
            for j, base_model in enumerate(self.base_model_list):
                new_fc = self._prep_backbone_and_new_fc(base_model, self.args.backbone_list[j], num_class)
                self.new_fc_list.append(new_fc)

        elif self.base_model_name is not None:
            self.new_fc = self._prep_backbone_and_new_fc(self.base_model, self.base_model_name, num_class)

    def forward(self, *argv, **kwargs):
        def backbone(input_data, the_base_model, new_fc):
            feat = the_base_model(input_data.view((-1, 3) + input_data.size()[-2:]))
            if new_fc is not None:
                base_out = new_fc(feat)
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            else:
                base_out = None
            feat = feat.view((-1, self.num_segments) + feat.size()[1:])
            return feat, base_out

        batch_size = kwargs["input"][0].shape[0] # TODO(yue) input[0] B*(TC)*H*W

        # TODO simple TSN
        if not self.args.ada_reso_skip:
            _, base_out = backbone(kwargs["input"][0], self.base_model, self.new_fc)
            output = self.consensus(base_out)
            return output.squeeze(1)

        input_list = kwargs["input"]
        feat_lite, _ = backbone(input_list[self.args.policy_input_offset], self.lite_backbone, None)
        hx = init_hidden(batch_size, self.hidden_dim)
        cx = init_hidden(batch_size, self.hidden_dim)
        logits_list = []
        lite_j_list = []
        feat_out_list = []
        base_out_list = []

        if self.multi_models:
            for bbi, the_backbone in enumerate(self.base_model_list):
                feat_out, base_out = backbone(input_list[bbi], the_backbone, self.new_fc_list[bbi])
                feat_out_list.append(feat_out)
                base_out_list.append(base_out)

        r_list=[]

        remain_skip_vector = torch.zeros(batch_size,1)
        old_hx = None
        old_r_t = None

        for t in range(self.num_segments):
            hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
            p_t = torch.log(F.softmax(self.linear(hx), dim=1))
            j_t = self.lite_fc(hx)
            logits_list.append(p_t) #TODO as logit
            lite_j_list.append(j_t) #TODO as pred

            #TODO (yue) need a simple case to illustrate this
            r_t_list=[]
            for b_i in range(p_t.shape[0]):
                r_t_item = F.gumbel_softmax(logits=p_t[b_i:b_i + 1], tau=kwargs["tau"], hard=True, eps=1e-10, dim=-1)
                r_t_list.append(r_t_item)

            r_t = torch.cat(r_t_list, dim=0)
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

        if self.args.policy_also_backbone:
            base_out_list.append(torch.stack(lite_j_list, dim=1))

        r_all = torch.stack(r_list, dim=1)
        output = self.combine_logits(r_all, base_out_list)

        return output.squeeze(1), r_all, None, torch.stack(base_out_list, dim=1)

    def combine_logits(self, r, base_out_list):
        # TODO r           N, T, K  (0-origin, 1-low, 2-skip)
        # TODO base_out_list  <K * (N, T, C)
        total_num = torch.sum(r[:,:, 0:self.reso_dim], dim=[1,2])
        res_base_list=[]
        for i,base_out in enumerate(base_out_list):
            r_like_base = r[:, :, i].unsqueeze(-1).expand_as(base_out) #TODO r[:,:,0] used to be, but wrong 01-30-2020
            res_base = torch.sum(r_like_base * base_out, dim=1)
            res_base_list.append(res_base)
        pred = torch.stack(res_base_list).sum(dim=0) / torch.clamp(total_num.unsqueeze(-1), 1)
        return pred

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


if __name__ == '__main__':
    #TODO load pretrained weights and make sure it works
    net = ARNet()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(10):
            y = net(x)
            print([y])