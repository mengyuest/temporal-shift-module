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
    }

prior_dict={
"efficientnet-b0": (0.39, 5.3),
"efficientnet-b1": (0.70, 7.8),
"efficientnet-b2": (1.00, 9.2),
"efficientnet-b3": (1.80, 12),
"efficientnet-b4": (4.20, 19),
"efficientnet-b5": (9.90, 30),
}

def get_gflops_params(model_name, resolution, num_classes, seg_len=-1, pretrained=True):
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
    else:
        exit("I don't know what is %s" % model_name)
    feat_dim = feat_dim_dict[model_name]
    setattr(model, last_layer, nn.Linear(feat_dim, num_classes))

    if seg_len == -1:
        dummy_data = torch.randn(1, 3, resolution, resolution)
    else:
        dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

    flops, params = profile(model, inputs=(dummy_data,))
    gflops = flops / 1e9
    params = params / 1e6

    return gflops, params


if __name__ == "__main__":

    #TODO(yue)
    # settings: resolution, number of classes, temporal dimension
    # we only compute one clip flops (one frame for 2Dconv, or one seg for 3Dconv)
    # 1. ResNet Family (resnet18,34,50,101)
    # 2. MobileNet V2
    # 3. EfficientNet Family (Eb0, Eb1, Eb3)
    # 4. ResNet3D Family (res3d18, res3d34, res3d50, res3d101)
    # 5. MobileNet3dV2

    str_list = []
    for k in [200]:
        for resolution in [84,112,168,224]:
            for key in feat_dim_dict:
                print(k,resolution,key)
                if "3d" in key:
                    seg_len = 16
                else:
                    seg_len = -1
                gflops, params = get_gflops_params(key, resolution, k, seg_len)
                str_list.append("%-25s\tclasses:%d\treso:%3d\tseg_len:%d\tgflops:%.4f\tparams:%.4fM" % (key, k, resolution, seg_len, gflops, params))

    for s in str_list:
        print(s)
