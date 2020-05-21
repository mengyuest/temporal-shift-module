import torch
import torch.nn as nn
import obsolete.ops.fresnet


class FNet(nn.Module):
    def __init__(self, arch, depth_list, shall_pretrain, args):
        super(FNet, self).__init__()

        self.base_model_list=nn.ModuleList()

        # TODO global dictionary for modules
        gds=[{} for _ in range(len(depth_list))]

        # TODO depth specific loading
        for i in range(len(depth_list)):
            self.base_model_list.append(getattr(obsolete.ops.fresnet, arch)(
                pretrained=(shall_pretrain and i==0), num_filters=depth_list[i]))
            for k,v in self.base_model_list[-1].named_modules():
                gds[i][k]=v

        # TODO sharing conv weights
        for i in range(len(depth_list)-1, 0, -1):
            for k in gds[i]:
                if "conv" in k or "downsample.0" in k:
                    #print(k,gds[0][k].weight.shape)
                    out_dim_total, in_dim_total = gds[0][k].weight.shape[:2]
                    if in_dim_total == 3:
                        in_channels = 3
                    else:
                        in_channels = depth_list[i] * in_dim_total // 64
                    out_channels = depth_list[i] * out_dim_total // 64
                    if depth_list[i]==64:
                        gds[i][k].weight = gds[0][k].weight
                    else:
                        del gds[i][k].weight
                        gds[i][k].weight = torch.nn.Parameter(gds[0][k].weight[:out_channels, :in_channels])

        # TODO loading pretrained model weights

    def forward(self, input, **kwargs):
        if "signal" not in kwargs:
            signal = 0
        else:
            signal = kwargs["signal"]
        return self.base_model_list[signal](input)


