import sys
sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from thop import profile
from obsolete.ops.cnn3d.i3d_resnet import i3d_resnet
import ops.batenet
import ops.cgnet
import ops.cg_utils


feat_dim_dict = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    'batenet18': 512,
    'batenet34': 512,
    'batenet50': 2048,
    'batenet101': 2048,
    'cgnet18': 512,
    'cgnet34': 512,
    'cgnet50': 2048,
    'cgnet101': 2048,
    "res3d18": 512,
    "res3d34": 512,
    "res3d50": 2048,
    "res3d101": 2048,
    "BNInception": 1024,
    "AdaBNInc": 1024,
    }

def get_gflops_params(model_name, resolution, num_classes, seg_len=-1, pretrained=True,args=None):
    last_layer = "fc"
    if "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained)
    elif "BNInception" in model_name:
        from archs.bn_inception import bninception
        model = bninception(args=args)
    elif "AdaBNInc" in model_name:
        from archs.bn_inception_ada import bninception_ada
        model = bninception_ada(args=args)
    elif "res3d" in model_name:
        model = i3d_resnet(int(model_name.split("res3d")[1]), pretrained=pretrained)
    elif "batenet" in model_name:
        model = getattr(ops.batenet, model_name)(pretrained=False, args=args)
    elif "cgnet" in model_name:
        model = getattr(ops.cgnet, model_name)(pretrained=False, args=args)
    else:
        exit("I don't know what is %s" % model_name)

    setattr(model, last_layer, nn.Linear(feat_dim_dict[model_name], num_classes))

    if seg_len == -1:
        dummy_data = torch.randn(1, 3, resolution, resolution)
        if "batenet" in model_name:
            dummy_data = torch.randn(1 * args.num_segments, 3, resolution, resolution)
        elif "AdaBNInc" in model_name:
            dummy_data = torch.randn(1 * args.num_segments, 3, resolution, resolution)
        elif "cgnet" in model_name:
            dummy_data = torch.randn(1, args.num_segments, 3, resolution, resolution)
    else:
        dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

    flops, params = profile(model, inputs=(dummy_data,))

    if args.shared_policy_net:
        args.shared_policy_net = False
        if "AdaBNInc" in model_name:
            from archs.bn_inception_ada import bninception_ada
            model = bninception_ada(args=args)
        else:
            model = getattr(ops.batenet, model_name)(pretrained=False, args=args)
        setattr(model, last_layer, nn.Linear(feat_dim_dict[model_name], num_classes))
        flops, _ = profile(model, inputs=(dummy_data,))
        args.shared_policy_net = True
    # sep_n=0
    # shared_n=0
    # rest_n=0
    # if args.shared_policy_net:
    #     for k,v in model.named_parameters():
    #
    #         if "gate_fc" in k:
    #             print(k)
    #             if "fc0s" not in k and "fc1s" not in k:
    #                 print(v.numel())
    #                 sep_n += v.numel()
    #             else:
    #                 shared_n += v.numel()
    #         else:
    #             rest_n+=v.numel()
    # print("total:  %.2f\npolicy: %.2f\nshared: %.2f\nrest:   %.2f"%(params/1e6, sep_n/1e6, shared_n/1e6, rest_n/1e6))
    # exit()
    flops = flops / dummy_data.shape[0]

    return flops / 1e9, params / 1e6

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

    # args = DebugClass()
    # args.threshold_loss_weight=0.0001
    # args.partitions=4
    # args.ginit=0.0
    # args.alpha=2.0
    # args.gtarget=1.0
    # args.use_group=True
    # args.shuffle=True
    # args.sparse_bp=True
    #
    # gflops, params = get_gflops_params("cgnet18", 224, 200, seg_len=-1, pretrained=False,
    #                                    num_filters_list=[], default_signal=-1, last_conv_same=False,
    #                                    msd_indices_list=[], mer_indices_list=[],
    #                                    args=args)


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




    args = DebugClass()
    args.ada_reso_skip = True
    args.gate_gumbel_softmax = True
    args.gate_history = True
    args.fusion_type = "cat"
    args.gate_hidden_dim = 1024
    args.relative_hidden_size = -1.0
    args.hidden_quota = -1
    args.shared_policy_net = True



    gflops, params = get_gflops_params("batenet50", 224, 174, seg_len=-1, pretrained=False,
                                       num_filters_list=[], default_signal=-1, last_conv_same=False,
                                       msd_indices_list=[], mer_indices_list=[],
                                       args=args)

    print("gflops", gflops, "params", params)


