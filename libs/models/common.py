# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import math
import torch
import torch.nn as nn


def build_norm_layer(type, num_features):
    norm = None
    if type == 'batch':
        norm = torch.nn.BatchNorm2d(num_features)
    elif type == 'instance':
        norm = torch.nn.InstanceNorm2d(num_features)

    return norm


def build_activate_layer(type):
    act = None
    if type == 'relu':
        act = torch.nn.ReLU(True)
    elif type == 'prelu':
        act = torch.nn.PReLU()
    elif type == 'lrelu':
        act = torch.nn.LeakyReLU(0.2, True)
    elif type == 'tanh':
        act = torch.nn.Tanh()
    elif type == 'sigmoid':
        act = torch.nn.Sigmoid()

    return act


class DeconvBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=True,
                 activation='prelu',
                 norm=None):
        super().__init__()
        self.deconv = self.deconv = torch.nn.ConvTranspose2d(input_channels,
                                                             output_channels,
                                                             kernel_size,
                                                             stride,
                                                             padding,
                                                             bias=bias)
        self.norm = build_norm_layer(norm, output_channels)
        self.activation = build_activate_layer(activation)

    def forward(self, x):
        f = self.deconv(x)
        if self.norm is not None:
            f = self.norm(f)
        out = self.activation(f) if self.activation is not None else f

        return out


class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 activation='prelu', norm=None):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

        self.activation = build_activate_layer(activation)
        self.norm = build_norm_layer(norm, output_channels)

    def forward(self, x):
        f = self.norm(self.conv(x)) if self.norm is not None else self.conv(x)
        out = self.activation(f) if self.activation is not None else f
        return out


def conv3x3(in_channels, out_channels, bias=True):
    return ConvBlock(in_channels, out_channels,
                     kernel_size=3, stride=1,
                     padding=1, bias=bias,
                     activation='', norm='')


class Upsampler(nn.Sequential):
    def __init__(self, scale, num_channels, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv3x3(num_channels, 4 * num_channels, bias=bias))
                # m.append(conv(num_channels, 2 * num_channels, 3, bias))
                # m.append(conv(2 * num_channels, 4 * num_channels, 3, bias))
                m.append(nn.PixelShuffle(upscale_factor=2))
                if bn:
                    m.append(nn.BatchNorm2d(num_channels))
                if act:
                    m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(conv3x3(num_channels, 9 * num_channels, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(num_channels))
            if act:
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
