#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from matplotlib import pyplot as plt
import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

class MsConv(nn.Module):
    def __init__(self, c1,c2):
        super().__init__()
        self.conv0 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        self.conv0_1 = nn.Conv2d(c1, c1, (1, 7), padding=(0, 3), groups=c1)
        self.conv0_2 = nn.Conv2d(c1, c1, (7, 1), padding=(3, 0), groups=c1)

        self.conv1_1 = nn.Conv2d(c1, c1, (1, 11), padding=(0, 5), groups=c1)
        self.conv1_2 = nn.Conv2d(c1, c1, (11, 1), padding=(5, 0), groups=c1)

        self.conv2_1 = nn.Conv2d(c1, c1, (1, 21), padding=(0, 10), groups=c1)
        self.conv2_2 = nn.Conv2d(c1, c1, (21, 1), padding=(10, 0), groups=c1)
        self.conv3 = nn.Conv2d(c1, c2, 1)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv0(x)
        x_0 = self.conv0_1(x)
        x_0 = self.conv0_2(x_0)
        x_1 = self.conv1_1(x)
        x_1 = self.conv1_2(x_1)
        x_2 = self.conv2_1(x)
        x_2 = self.conv2_2(x_2)
        x_00 = x+x_0+x_1+x_2
        # print(f"x_00: {x_00.shape}")
        x = self.bn(self.conv3(x_00))
        # print(f"x: {x.shape}")
        return x


class SiLU(nn.Module):
	# SiLU激活函数
	@staticmethod
	def forward(x):
		return x * torch.sigmoid(x)
class ConvSNP(nn.Module):
    # 卷积SNP
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c1, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))
        # print(f"Input shape: {x.shape}")  # 调试代码
        # return self.act(self.bn(self.conv(x)))
        # return self.act(self.conv(self.bn(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class PCDB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c1 * e)
        self.Channelconfusion = ShuffleAttentionV1.channel_shuffle
        self.PConV1 = PConv(c1, c_)
        self.Conv1 = Conv_1(2*c_, c_)
        self.Spatial_attention = ShuffleAttentionV1(channel=c1,forward_state='forward_spatial')
        self.Channel_attention = ShuffleAttentionV1(channel=c1,forward_state='forward_channel')
        self.bn = nn.BatchNorm2d(c_)
        self.relu = nn.ReLU(inplace=True)
        self.pconv2 =PConv(c_, c2)
        self.add = shortcut and c1==c2
    def forward(self, x):
        b, c, h, w =x.size()

        x1 = x
        # print(f"x_shape{x.shape}")
        x1 = self.Channelconfusion(x,2)
        x1 = self.PConV1(x1)
        # print(f"x1_shape{x1.shape}")
        x2 = torch.cat([x1, x], dim=1 )
        # print(f"x2_shape{x2.shape}")
        x2 = self.Conv1(x2)
        # print(f"x2_shape{x2.shape}")
        # x2 = self.conv2(x2)
        x3 = x2.view(b * 8, -1, h, w)
        # print(f"x3_shape{x3.shape}")
        x_0, _ = x3.chunk(2, dim=1)
        # print(f"x0_shape{x_0.shape}")
        x_spatial = self.Spatial_attention(x2)
        x_Channel = self.Channel_attention(x2)
        x4 = torch.cat([x_spatial,x_Channel], dim=1 )
        # x4 = (x_0*x_Channel)*x_spatial
        # print(f"x4_shape{x4.shape}")
        # x5 = torch.cat([x4,x_0], dim=1)
        # print(f"x5_shape{x5.shape}")
        out = x4.contiguous().view(b, -1, h, w)
        # print(f"out_shape{out.shape}")
        out = self.Channelconfusion(out,2)
        # print(f"out_shape{out.shape}")
        out = self.relu(self.bn(out))
        # print(f"out_shape{out.shape}")
        if self.add:
            out = x + self.pconv2(out)
        else:
            out = self.pconv2(out)
        return out

class ShuffleAttentionV1(nn.Module):
    def __init__(self, forward_state, channel=512, reduction=16, G=8, n_div=4):
        super(ShuffleAttentionV1, self).__init__()
        self.n_div = n_div
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.cbias = Parameter(torch.ones(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sweight = Parameter(torch.zeros(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sbias = Parameter(torch.ones(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        if forward_state == 'forward_channel':
            self.forward = self.forward_channel
        elif forward_state == 'forward_spatial':
            self.forward = self.forward_spatial
        else:
            raise NotImplementedError

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward_channel(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)
        x_0, _ = x.chunk(2, dim=1)

        # channel attention
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        # x_channel = self.sigmoid(x_channel)
        return x_channel

    def forward_spatial(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)
        _, x_1 = x.chunk(2, dim=1)

        # spatial attention
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        # x_spatial = self.sigmoid(x_spatial)
        return x_spatial


class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, type_forward='split_cat'):
        """
        Partial Convolution
        :param dim: Number of input channels
        :param ouc: Output channels
        :param n_div: Reciprocal of the partial ratio
        :param forward: Forward type, 'slicing' or 'split_cat'
        """
        super(PConv, self).__init__()
        self.type_forward = type_forward
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(dim, ouc, kernel_size=1)
        # self.conv = nn.Conv2d(dim, ouc, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        if type_forward == 'slicing':
            self.forward = self.forward_slicing
        elif type_forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(
            x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # 将 x 通道分为两部分，一部分为 in_channel//n_div, 另一部分为 in_channel - in_channel//n_div
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 对 x1 部分卷积 --> in_channel = in_channel // n_div; output_channel = in_channel // n_div --> 保持通道数不变
        x1 = self.partial_conv3(x1)
        # 在通道上拼接 x1, x2
        x = torch.cat((x1, x2), 1)
        # 对 x 做一个卷积，改变通道数
        x = self.conv(x)
        return x

class Conv_1(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super(Conv_1, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k,
                              stride=s, groups=g, bias=False)

    def forward(self, x):
        return self.conv(x)


class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(
            2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(
            3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(
            N, self.kernel_size ** 2, H, W, self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(
            0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)
        in_tensor = in_tensor.unfold(
            3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(
            in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(
            c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, n=1, e=0.5, reduction_ratio=16):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
 
        self.c = int(out_planes * e)
        self.cv1 = Conv_1(in_planes, 2 * self.c, 1, 1)
        self.cv2 = Conv_1((2 + n) * self.c, out_planes, 1)
        self.cv3 = Conv_1(in_planes, out_planes, 1, 1)
        self.m = nn.ModuleList(Bottleneck(
            self.out_planes // 2, self.out_planes // 2) for _ in range(n))

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.rate3 = torch.nn.Parameter(torch.Tensor(1))

        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 3, padding=1)
        self.conv_key = Conv_1(in_planes, 1)

        self.softmax = torch.nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            Conv_1(self.in_planes, self.out_planes, 1),
            nn.LayerNorm([self.out_planes, 1, 1]),
            nn.ReLU(),
            Conv_1(self.out_planes, self.in_planes, 1),
        )
        self.change_channels = Conv_1(self.in_planes, self.out_planes, 1)
        # self.conv4_gobal = nn.Conv2d(in_planes, 1, kernel_size=1, stride=1)
        # self.sigomid = nn.Sigmoid()
        # for group_id in range(0, 4):
        #     self.interact = nn.Conv2d(in_planes // 4, in_planes // 4, 1, 1, )
        # self.group_num = 16
        # self.eps = 1e-10
        # self.gamma = nn.Parameter(torch.randn(in_planes, 1, 1))
        # self.beta = nn.Parameter(torch.zeros(in_planes, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
			nn.Linear(out_planes, out_planes // reduction_ratio, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(out_planes // reduction_ratio, out_planes, bias=False),
		)
        self.sigmoid = nn.Sigmoid()	
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        init_rate_half(self.rate3)

    def forward(self, x):
        k_trans = self.conv_key(x)
        b, c, h, w = x.shape
        # print(f"x_shape{x.shape}")

        key = self.softmax(k_trans.view(
            b, 1, -1).permute(0, 2, 1).contiguous())
        query = x.view(b, -1, h * w)
        # b, c, h*w matmul b, h*w, 1 --> b, c, 1
        concate_QK = torch.matmul(query, key).view(b, c, 1, 1).contiguous()
        out_att = self.change_channels(x + self.Conv_value(concate_QK))

        # conv
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out_conv = self.cv2(torch.cat(y, 1))
		# channel att
        z = self.cv3(x)
        avg_out = self.fc(self.avg_pool(z).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(z).view(x.size(0), -1))
        out_channel = avg_out + max_out
        # # print(f"out_channel_shape{out_channel.shape}")
        channel_weights = self.sigmoid(out_channel).view(z.size(0), z.size(1), 1, 1)
        x_channel = z * channel_weights
        # print(f"channel_shape{x_channel.shape}")
		

        return self.rate1 * out_att + self.rate2 * out_conv 
        # return out_conv




def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mh, Mw, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # pad是从后往前，从左往右，从上往下，原顺序是（B,C,H,W) pad顺序就是(W，H，C）
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W  # 这里是经过padding的H和W


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 将通道数由4倍变为2倍

    def forward(self, x, H, W):
        """
        x: B, H*W（L）, C，并不知道H和W，所以需要单独传参
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 因为是下采样两倍，如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 此时（B,H,W,C)依然是从后向前
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C],这里的-1就是在C的维度上拼接
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 每一个head都有自己的relative_position_bias_table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # meshgrid生成网格，再通过stack方法拼接
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(
            1)  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # 整个训练当中，window_size大小不变，因此这个索引也不会改变
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多头融合
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # 通过unbind分别获得qkv
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        # 通过unsqueeze加上一个batch维度
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        B, C, H, W = x.shape
        L = H * W
        x = x.view(B, C, L).transpose(1, 2)  # (B, L, C)
        # 残差网络
        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, H, W, C)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # 对窗口进行移位。从上向下移，从左往右移，因此是负的
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # shift还原，如果没有shifted就不用还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 将输出形状从 [B, L, C] 转换为 [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 移动尺寸

        # 在当前stage之中所有的block
        # 注意每个block中只会有一个MSA,要么W-MSA，要么SW-MSA，所以shift_size为0代表W-MSA，不为0代表SW-MSA
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # 将窗口切分，然后进行标号
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1] 划为窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw] 窗口展平
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # 先创建一个mask蒙版，在图像尺寸不变的情况下蒙版也不改变
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # 默认不适用checkpoint方法
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            # 防止H和W是奇数。如果是奇数，在下采样中经过一次padding就变成偶数了，但如果这里不给H和W加一的话就会导致少一个，如果是偶数，加一除二取整还是不变
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W
class SwinTransformer(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        # 对应Patch partition和Linear Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 在每个block的dropout率，是一个递增序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # num_layers及stage数
            # 与论文不同，代码中的stage包含的是下一层的Patch merging ，因此在最后一个stage中没有Patch merging
            # dim为当前stage的维度，depth是当前stage堆叠多少个block，drop_patch是本层所有block的drop_patch
            # downsample是Patch merging，并且在最后一个stage为None
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 在这个分类任务中，用全局平均池化取代cls token
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # 依次通过每个stage
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    model = MsConv(model)
    return model


swin_tiny_patch4_window7_224()