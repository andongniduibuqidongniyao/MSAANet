import torch
from main_structure.model import *
from torch import nn

from main_structure.model import PConv, ShuffleAttentionV1, Conv_1
from main_structure.GPM import SwinTransformerBlock

def autopad(k, p=None, d=1):
	# kernel, padding, dilation
	if d > 1:
		# actual kernel-size
		k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
	if p is None:
		# auto-pad
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
	return p


class SiLU(nn.Module):
	# SiLU激活函数
	@staticmethod
	def forward(x):
		return x * torch.sigmoid(x)


class Conv(nn.Module):
	default_act = SiLU()
	
	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
		super().__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
		self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
		self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
	
	def forward(self, x):
		# print(f"Input shape: {x.shape}")
		return self.act(self.bn(self.conv(x)))
	
	def forward_fuse(self, x):
		return self.act(self.conv(x))

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
class C2f(nn.Module):
	def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
		super().__init__()
		self.c = int(c2 * e)
		# self.cv1 = Conv(c1, 2 * self.c, 1, 1)
		self.cv1 = Conv_1(c1, 2 * self.c, 1, 1)
		# self.cv2 = Conv((2 + n) * self.c, c2, 1)
		self.cv2 = Conv_1((2 + n) * self.c, c2, 1)
		# self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
		self.m = nn.ModuleList(PCDB(self.c, self.c, shortcut, e=1.0) for _ in range(n))
	
	def forward(self, x):
		y = list(self.cv1(x).split((self.c, self.c), 1))
		y.extend(m(y[-1]) for m in self.m)
		return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
	def __init__(self, c1, c2, k=5):
		super().__init__()
		c_ = c1 // 2
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c_ * 4, c2, 1, 1)
		self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
	
	def forward(self, x):
		x = self.cv1(x)
		y1 = self.m(x)
		y2 = self.m(y1)
		return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))





class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.GPM = SwinTransformerBlock(
            dim=base_channels,  # 输入特征的维度（与 dark2 的输出通道数匹配）
            num_heads=8,  # 注意力头的数量
            window_size=7,  # 窗口大小
            shift_size=0  # 移位大小（确保 0 <= shift_size < window_size）
        )
        self.dark2 = nn.Sequential(

            # Conv(base_channels, base_channels * 2, 3, 2),
            # MsConv(base_channels, base_channels),
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            # Conv(base_channels * 2, base_channels * 4, 3, 2),
            MsConv(base_channels * 2, base_channels * 2),
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            # Conv(base_channels * 4, base_channels * 8, 3, 2),
            MsConv(base_channels * 4, base_channels * 4),
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            # Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            MsConv(base_channels * 8, base_channels * 8),
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )

        if pretrained:
            url = {
                "n": 'SourceFile/pretrained/yolov8_l_backbone_weights.pth',
                "s": 'SourceFile/pretrained/yolov8_s_backbone_weights.pth',
                "m": 'SourceFile/pretrained/yolov8_m_backbone_weights.pth',
                "l": 'SourceFile/pretrained/yolov8_l_backbone_weights.pth',
                "x": 'SourceFile/pretrained/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./Category_Files")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])            
    def forward(self, x):
        x = self.stem(x)
        x = self.GPM(x, attn_mask=None)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3



