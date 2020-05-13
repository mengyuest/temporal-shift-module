import sys
sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from thop import profile
from efficientnet_pytorch import EfficientNet
from ops.cnn3d.i3d_resnet import i3d_resnet
from ops.cnn3d.mobilenet3dv2 import mobilenet3d_v2
from ops.cnn3d.shufflenet3d import ShuffleNet3D
import ops.dmynet
import ops.dhsnet
import ops.gatenet
import ops.cgnet
import ops.cg_utils
import ops.msdnet
import ops.mernet

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
    "shufflenet3d_0.5": 480,
    "shufflenet3d_1.0": 960,
    "shufflenet3d_1.5": 1440,
    "shufflenet3d_2.0": 1920,
    "mobilenet3dv2": 1280,
    "res3d18": 512,
    "res3d34": 512,
    "res3d50": 2048,
    "res3d101": 2048,
    "dmynet18": 512,
    "dmynet34": 512,
    "dmynet50": 2048,
    "dmynet101": 2048,
    'dhsnet18': 512,
    'dhsnet34': 512,
    'dhsnet50': 2048,
    'dhsnet101': 2048,
    'gatenet18': 512,
    'gatenet34': 512,
    'gatenet50': 2048,
    'gatenet101': 2048,
    'cgnet18': 512,
    'cgnet34': 512,
    'cgnet50': 2048,
    'cgnet101': 2048,
    "msdnet": 0,
    "mernet50": 0,
    "ir_csn_50": 2048,
    "ir_csn_152": 2048,
    }

prior_dict={
"efficientnet-b0": (0.39, 5.3),
"efficientnet-b1": (0.70, 7.8),
"efficientnet-b2": (1.00, 9.2),
"efficientnet-b3": (1.80, 12),
"efficientnet-b4": (4.20, 19),
"efficientnet-b5": (9.90, 30),
}

def get_gflops_params(model_name, resolution, num_classes, seg_len=-1, pretrained=True,
                      num_filters_list=[], default_signal=-1, last_conv_same=False,
                      msd_indices_list=[], mer_indices_list=[], args=None):
    if model_name in prior_dict:
        gflops, params = prior_dict[model_name]
        gflops = gflops / 224 / 224 * resolution * resolution
        return gflops, params

    if "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained)
        last_layer = "fc"
    elif model_name == "mobilenet_v2":
        model= getattr(torchvision.models, model_name)(pretrained)
        last_layer = "classifier"
    # elif "efficientnet-" in model_name:
    #     if pretrained:
    #         model= EfficientNet.from_pretrained(model_name)
    #     else:
    #         model = EfficientNet.from_named(model_name)
    #     last_layer = "_fc"
    elif "shufflenet3d" in model_name:
        model = ShuffleNet3D(3, width_mult=float(model_name.split("_")[1]))
        last_layer = "classifier"
    elif "res3d" in model_name:
        model = i3d_resnet(int(model_name.split("res3d")[1]), pretrained=pretrained)
        last_layer = "fc"
    elif "mobilenet3d" in model_name:
        model = mobilenet3d_v2(pretrained=pretrained)
        last_layer = "classifier"
    elif "dmynet" in model_name:
        model = getattr(ops.dmynet, model_name)(pretrained=False,
                                                num_filters_list=num_filters_list,
                                                default_signal=default_signal)
        last_layer = "fcs"
    elif "dhsnet" in model_name:
        model = getattr(ops.dhsnet, model_name)(pretrained=False,
                                                num_filters_list=num_filters_list,
                                                default_signal=default_signal,
                                                args=args)
        last_layer="fc"
    elif "gatenet" in model_name:
        model = getattr(ops.gatenet, model_name)(pretrained=False, args=args)
        last_layer = "fc"
        # print(model.count_flops((1, 1, 3, 224, 224)))
    elif "cgnet" in model_name:
        model = getattr(ops.cgnet, model_name)(pretrained=False, args=args)
        last_layer = "fc"
    elif "msdnet" in model_name:
        model = getattr(ops.msdnet, "MSDNet")(default_signal = default_signal)
    elif "mernet" in model_name:
        model = getattr(ops.mernet, model_name)(default_signal = default_signal)
    else:
        exit("I don't know what is %s" % model_name)
    feat_dim = feat_dim_dict[model_name]
    if "dmynet" in model_name:
        if last_conv_same:
            setattr(model, last_layer, torch.nn.ModuleList(
                [nn.Linear(feat_dim, num_classes) for _ in num_filters_list]))
        else:
            setattr(model, last_layer, torch.nn.ModuleList([nn.Linear(feat_dim * num_filters // 64, num_classes) for num_filters in num_filters_list]))
    elif "msdnet" in model_name:
        for msd_i in msd_indices_list:
            msd_feat_dim = model.classifier[msd_i].linear.in_features
            model.classifier[msd_i].linear = nn.Linear(msd_feat_dim, num_classes)
    elif "mernet" in model_name:
        for mer_i in mer_indices_list:
            mer_feat_dim = getattr(model, "fc%d.in_features"%(mer_i))
            setattr(model, "fc%d"%mer_i, nn.Linear(mer_feat_dim, num_classes))
    else:  # TODO: normal, dhs, gate
        setattr(model, last_layer, nn.Linear(feat_dim, num_classes))

    if seg_len == -1:
        dummy_data = torch.randn(1, 3, resolution, resolution)
    else:
        dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)



    hooks={}
    if "dmynet" in model_name:
        hooks = {ops.dmynet.Conv2dMY: ops.dmynet.count_conv_my}
    if "dhsnet" in model_name or "gatenet" in model_name or "cgnet" in model_name:  # TODO: step-by-step solution
        dummy_data = torch.randn(2, 8, 3, resolution, resolution)
        if "dhsnet" in model_name:
            hooks = {ops.dhsnet.Conv2dHS: ops.dhsnet.count_conv_hs}
        if "cgnet" in model_name:
            hooks = {ops.cg_utils.CGConv2dNew: ops.cg_utils.count_cg_conv2d}

            # print(model)
    # print(model.layer1[0].conv1)
    # print(type(model.layer1[0].conv1))
    flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    # for k,m in model.named_modules():
    #     if isinstance(m, torch.nn.Conv2d):
    #         print(k, m.total_ops)
    if "dhsnet" in model_name or "gatenet" in model_name or "cgnet" in model_name:
        flops = flops / 16
    gflops = flops / 1e9
    params = params / 1e6

    return gflops, params

class DebugClass(object):
    pass


if __name__ == "__main__":

    #TODO(yue)
    # settings: resolution, number of classes, temporal dimension
    # we only compute one clip flops (one frame for 2Dconv, or one seg for 3Dconv)
    # 1. ResNet Family (resnet18,34,50,101)
    # 2. MobileNet V2
    # 3. EfficientNet Family (Eb0, Eb1, Eb3)
    # 4. ResNet3D Family (res3d18, res3d34, res3d50, res3d101)
    # 5. MobileNet3dV2

    # str_list = []
    # for k in [200]:
    #     for resolution in [84,112,168,224]:
    #         for key in feat_dim_dict:
    #             print(k,resolution,key)
    #             if "3d" in key:
    #                 seg_len = 16
    #             else:
    #                 seg_len = -1
    #             gflops, params = get_gflops_params(key, resolution, k, seg_len)
    #             str_list.append("%-25s\tclasses:%d\treso:%3d\tseg_len:%d\tgflops:%.4f\tparams:%.4fM" % (key, k, resolution, seg_len, gflops, params))
    #
    # for s in str_list:
    #     print(s)

    # str_list = []
    # k=200
    # seg_len=-1
    # for resolution in [84, 112, 168, 224]:
    #     for signal in [0,1,2,3]:
    #         base_model_gflops, params = get_gflops_params("dmynet50", resolution, k, seg_len,
    #                                                       num_filters_list=[64,48,32,16], default_signal=signal)
    #         str_list.append("%-25s\tclasses:%d\treso:%3d\tseg_len:%d\tgflops:%.4f\tparams:%.4fM" % (
    #                 "dmynet50", k, resolution, seg_len, base_model_gflops, params))
    #
    # for s in str_list:
    #     print(s)

    #TODO debug

    # gflops, params = get_gflops_params("dmynet50", 224, 200, seg_len=-1, pretrained=False,
    #                   num_filters_list=[64, 48, 32], default_signal=0, last_conv_same=False,
    #                   msd_indices_list=[], mer_indices_list=[], args=None)


    args = DebugClass()
    args.threshold_loss_weight=0.0001
    args.partitions=4
    args.ginit=0.0
    args.alpha=2.0
    args.gtarget=1.0
    args.use_group=True
    args.shuffle=True
    args.sparse_bp=True

    gflops, params = get_gflops_params("cgnet18", 224, 200, seg_len=-1, pretrained=False,
                                       num_filters_list=[], default_signal=-1, last_conv_same=False,
                                       msd_indices_list=[], mer_indices_list=[],
                                       args=args)


    # gflops, params = get_gflops_params("resnet18", 224, 200, seg_len=-1, pretrained=False,
    #                                    num_filters_list=[], default_signal=-1, last_conv_same=False,
    #                                    msd_indices_list=[], mer_indices_list=[],
    #                                    args=args)


    # import ops.cg_utils as G
    #
    # conv1 = G.CGConv2d(3, 64, kernel_size=3,
    #                         stride=1, padding=1, bias=False,
    #                         p=args.partitions, th=args.ginit, alpha=args.alpha,
    #                         use_group=args.use_group, shuffle=args.shuffle, sparse_bp=args.sparse_bp)
    #
    # print(conv1)
    # print(type(conv1))
    print("gflops", gflops, "params", params)


