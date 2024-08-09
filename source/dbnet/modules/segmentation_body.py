# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 10:29
# @Author  : zhoujun
import torch
import torch.nn.functional as F
from torch import nn

from .basic import ConvBnRelu

class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_feature, init_weight=True):
        super().__init__()
        
        self.spatial_wise = nn.Sequential(
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_feature, 1, bias=False),
            nn.Sigmoid()
        )
        
        if init_weight:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        new_x = torch.mean(x, dim=1, keepdim=True)
        new_x = self.spatial_wise(new_x) + x 
        new_x = self.attention_wise(new_x)
        return new_x
    
class AdaptiveScaleFusion(nn.Module):
    def __init__(self, in_channels=256, inter_channels=64, out_features=4):
        super().__init__()
        
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features = out_features
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
            
    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features    
        x = []

        for i in range(self.out_features):
            x.append(score[:, i:i+1] * features_list[i])
            
        return torch.cat(x, dim=1)
        
    
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.kernel_size = kernel_size 

        self.conv2d = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, inputs):
        avg_pool = torch.mean(inputs, dim=1, keepdim=True)
        max_pool, _ = torch.max(inputs, dim=1, keepdim=True)

        x = torch.cat([avg_pool, max_pool], dim=1)
#         x = avg_pool
        x = self.conv2d(x)
        x = self.sigmoid(x)

        return x
    
class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.block = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.block(self.avg_pool(x))
        max_out = self.block(self.max_pool(x))
        out = avg_out + max_out
        
        return self.sigmoid(out)

class FPN(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256, use_spatial_attention=True):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        
        self.use_spatial_attention = use_spatial_attention
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out,
                      self.conv_out,
                      kernel_size=3,
                      padding=1,
                      stride=1), nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace))
        
#         self.spatial_attention_2 = SpatialAttention()
#         self.spatial_attention_3 = SpatialAttention()
#         self.spatial_attention_4 = SpatialAttention()
#         self.spatial_attention_5 = SpatialAttention()
        self.spatial_attention = AdaptiveScaleFusion()
        
#         self.channel_attention = ChannelAttention(self.conv_out)
        
        self.out_channels = self.conv_out
        
        self.conv.apply(self.weights_init)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        
#         if self.use_spatial_attention:
# #             p2 = self.spatial_attention_2(p2) * p2
# #             p3 = self.spatial_attention_3(p3) * p3
# #             p4 = self.spatial_attention_4(p4) * p4
# #             p5 = self.spatial_attention_5(p5) * p5
            
        x, p2, p3, p4, p5 = self._upsample_cat(p2, p3, p4, p5)

        if self.use_spatial_attention:
            x = self.spatial_attention(x, [p2, p3, p4, p5])
            
#         x = self.conv(x)
        
#         x = self.channel_attention(x) * x
        
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1), p2, p3, p4, p5
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
            
class _FPN(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels,
                                    inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out,
                      self.conv_out,
                      kernel_size=3,
                      padding=1,
                      stride=1), nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace))
        
        self.spatial_attention_2 = SpatialAttention()
        self.spatial_attention_3 = SpatialAttention()
        self.spatial_attention_4 = SpatialAttention()
        self.spatial_attention_5 = SpatialAttention()
#         self.spatial_attention = AdaptiveScaleFusion()
        
        self.channel_attention = ChannelAttention(self.conv_out)
        
        self.out_channels = self.conv_out
        
#         self.conv.apply(self.weights_init)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        
        x, p2, p3, p4, p5 = self._upsample_cat(p2, p3, p4, p5)
            
        x = self.conv(x)

        
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1), p2, p3, p4, p5
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


class FPEM_FFM(nn.Module):
    def __init__(self,
                 backbone_out_channels,
                 inner_channels=128,
                 fpem_repeat=2, use_spatial_attention=False):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        self.conv_out = inner_channels
        inplace = True
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(self.conv_out))
        self.out_channels = self.conv_out * 4

    def forward(self, x):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:])
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:])
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:])
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        return Fy


class FPEM(nn.Module):
    def __init__(self, in_channels=128, use_spatial_attention=False):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # down 阶段
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=stride,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
