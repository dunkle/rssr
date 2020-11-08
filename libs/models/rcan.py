# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import math

import torch
import torch.nn as nn

from .common import conv3x3, Upsampler



# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, num_channels, reduction,
                 bn=False, res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv3x3(num_channels, num_channels))
            if bn:
                modules_body.append(nn.BatchNorm2d(num_channels))
            if i == 0:
                modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(CALayer(num_channels, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, num_channels, reduction, res_scale, num_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(num_channels,
                             reduction,
                             bn=False,
                             res_scale=res_scale) for _ in range(num_resblocks)]
        modules_body.append(conv3x3(num_channels, num_channels))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res





## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self,
                 num_resgroups,
                 num_resblocks,
                 channels,
                 scale,
                 reduction,
                 num_colors=3):
        super(RCAN, self).__init__()

        # define head module
        modules_head = [conv3x3(3, channels)]

        # define body module
        modules_body = [
            ResidualGroup(num_channels=channels,
                          reduction=reduction,
                          res_scale=1,
                          num_resblocks=num_resblocks) for _ in range(num_resgroups)]

        modules_body.append(conv3x3(channels, channels))

        # define tail module
        modules_tail = [
            Upsampler(scale=scale, num_channels=channels),
            conv3x3(in_channels=channels, out_channels=num_colors)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


def build_model(cfgs):
    model_cfg = cfgs.model
    model = RCAN(num_resgroups=model_cfg.num_resgroups,
                 num_resblocks=model_cfg.num_resblocks,
                 channels=model_cfg.channels,
                 scale=cfgs.data.scale,
                 reduction=model_cfg.reduction)

    if model_cfg.pretrained is not None:
        model.load_state_dict(torch.load(model_cfg.pretrained))

    return model
