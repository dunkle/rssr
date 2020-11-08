# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import torch.nn as nn

from .common import DeconvBlock, ConvBlock, conv3x3


class Upsampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class DwonProjectionUnit(nn.Module):

    def __init__(self, channels, kernel_size=4, stride=4, padding=2, bias=True, activation='prelu', norm=''):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=kernel_size,
                               bias=bias, stride=stride, padding=padding,
                               activation=activation, norm=norm)
        self.deconv = DeconvBlock(channels, channels, kernel_size=kernel_size,
                                  bias=bias, stride=stride, padding=padding,
                                  activation=activation, norm=norm)
        self.conv2 = ConvBlock(channels, channels, kernel_size=kernel_size,
                               bias=bias, stride=stride, padding=padding,
                               activation=activation, norm=norm)

    def forward(self, x):
        l0 = self.conv1(x)
        h0 = self.deconv(l0)
        l1 = self.conv2(h0 - x)
        out = l1 + l0
        return out


class UpProjectionUnit(nn.Module):
    """
    """

    def __init__(self, channels, kernel_size=4, stride=4, padding=2, bias=True, activation='prelu', norm=''):
        super().__init__()
        self.deconv1 = DeconvBlock(channels, channels, kernel_size=kernel_size,
                                   bias=bias, stride=stride, padding=padding,
                                   activation=activation, norm=norm)
        self.conv = ConvBlock(channels, channels, kernel_size=kernel_size,
                              bias=bias, stride=stride, padding=padding,
                              activation=activation, norm=norm)
        self.deconv2 = DeconvBlock(channels, channels, kernel_size=kernel_size,
                                   bias=bias, stride=stride, padding=padding,
                                   activation=activation, norm=norm)

    def forward(self, x):
        h0 = self.deconv1(x)
        l0 = self.conv(h0)
        h1 = self.deconv2(l0 - x)
        out = h1 + h0
        return out


class RDBPBlock(nn.Module):
    """
    Residual Dense Back-projection block
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RDBN(nn.Module):

    def __init__(self, scale, channels, num_colors=3):
        super().__init__()
        self.head = conv3x3(3, channels)

        self.body = RDBPBlock()

        self.tail = nn.Sequential(
            Upsampler(scale=scale, num_channels=channels),
            conv3x3(in_channels=channels, out_channels=num_colors)
        )

    def forward(self, x):
        f0 = self.head(x)
        fb = self.body(f0)
        flr = f0 + fb
        out = self.tail(flr)

        return out
