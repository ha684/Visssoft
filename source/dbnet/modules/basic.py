# -*- coding: utf-8 -*-
# @Time    : 2019/12/6 11:19
# @Author  : zhoujun
from torch import nn


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 inplace=True):
        super().__init__()
        
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               dilation=dilation,
#                               groups=groups,
#                               bias=bias,
#                               padding_mode=padding_mode),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=inplace)
#         )
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

#         self.block.apply(self.weights_init)

    def forward(self, x):
#         x = self.block(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
