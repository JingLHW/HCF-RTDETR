import torch
import torch.nn as nn
from collections import OrderedDict
import math
import torch.nn.functional as F

class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

def get_activation(act: str, inpace: bool = True):
    act = act.lower() if isinstance(act, str) else act

    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f'Unsupported activation: {act}')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvNormLayer(nn.Module):
    """
    Conv2d -> BN -> Activation
    """
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    """
    Standard convolution: Conv2d -> BN -> Activation
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True
            else act if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)
        return out


# =========================================================
# PConv branch
# =========================================================

class PConv(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3

        self.conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(f'Unsupported forward type: {forward}')

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        return x


class PConv_bottleneck(nn.Module):
    """
    Conv 1x1 -> PConv 3x3 -> Conv 1x1 -> BN -> Add(identity)
    """
    def __init__(self, channels, act='relu'):
        super().__init__()

        self.conv1 = ConvNormLayer(
            channels, channels, kernel_size=1, stride=1, act=act
        )
        self.pconv = PConv(dim=channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.pconv(out)
        out = self.conv2(out)
        out = self.bn(out)

        out = out + identity
        return out


class PConv_Block(nn.Module):
    """
    Conv 3x3 -> BN -> ReLU -> PConv_bottleneck -> BN -> Add(shortcut) -> ReLU
    """
    def __init__(self, ch_in, ch_out, stride=1, shortcut=True, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut
        self.act = get_activation(act)

        self.conv1 = nn.Conv2d(
            ch_in, ch_out,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = get_activation(act)

        self.pconv_bottleneck = PConv_bottleneck(ch_out, act=act)
        self.bn2 = nn.BatchNorm2d(ch_out)

        if not shortcut or ch_in != ch_out or stride != 1:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(ch_out)
                )
            else:
                self.short = nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(ch_out)
                )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        identity = self.short(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pconv_bottleneck(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)
        return out


# =========================================================
# SFDConv / FCB branch
# =========================================================

class DSConvCore(nn.Module):
    def __init__(self, ch_in, ch_out, act=None):
        super().__init__()
        self.proj = ConvNormLayer(ch_in, ch_out, kernel_size=3, stride=1, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x


class DSConv(nn.Module):
    def __init__(self, ch_in, ch_out, act=None):
        super().__init__()
        self.block = DSConvCore(ch_in, ch_out, act=act)

    def forward(self, x):
        return self.block(x)


class SpectralOperator(nn.Module):
    """
    Learnable spectral operator.
    """
    def __init__(self, channels, size=None, stride=1):
        super().__init__()
        self.channels = channels
        self.size = size
        self.stride = stride

        kernel_size = 1 if stride == 1 else 3
        self.op = Conv(channels, channels, k=kernel_size, s=stride)

    def forward(self, x):
        return self.op(x)


class FCB(nn.Module):
    """
    FCB module:
    Channel-wise Split(freq_ratio)
    -> spectral branch
    -> spatial identity branch
    -> concat
    -> Conv1x1 -> BN -> ReLU
    """
    def __init__(
        self,
        c: int,
        size=None,
        s: int = 1,
        freq_ratio: float = 0.15,
        min_freq_ch: int = 8,
        gamma_init: float = 0.3,
    ):
        super().__init__()

        c_freq = max(int(round(c * freq_ratio)), 0)
        if c_freq < min_freq_ch:
            c_freq = 0
        c_spat = c - c_freq

        self.split = (c_spat, c_freq)
        self.has_freq = c_freq > 0
        self.size = size

        if self.has_freq:
            self.spectral_op = SpectralOperator(c_freq, size=size, stride=s)

        self.conv1x1 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        if not self.has_freq:
            y = self.conv1x1(x)
            y = self.bn(y)
            y = self.act(y)
            return y

        x_spat, x_freq = torch.split(x, self.split, dim=1)

        y_freq = self.spectral_op(x_freq)
        y_freq = self.gamma * y_freq

        y = torch.cat([x_spat, y_freq], dim=1)
        y = self.conv1x1(y)
        y = self.bn(y)
        y = self.act(y)

        return y


class SFDConv(BasicBlock):
    """
    DSConv -> FCB
    """
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)

        self.branch2b = nn.Sequential(
            DSConv(ch_out, ch_out, act=nn.ReLU()),
            FCB(ch_out)
        )



class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)



class VoVGSCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        c1: 输入通道（由 parse_model 用 ch[from] 传进来）
        c2: 输出通道（YAML 里写的那个 256）
        n : GSBottleneck 重复次数（YAML 里的那个 3）
        e : hidden ratio（YAML 里写的 0.5）
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"VoVGSCSP: expect {self.c1} channels, but got {x.shape[1]}"
            )

        x1 = self.gsb(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))




class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)


class CCFM(nn.Module):
    def __init__(self, c3=128, c4=256, c5=256, out_c=256):
        super().__init__()

        # project
        self.p3_proj = GSConv(c3, out_c, 1, 1)
        self.p4_proj = GSConv(c4, out_c, 1, 1)
        self.p5_proj = GSConv(c5, out_c, 1, 1)

        # top-down
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p4 = Fusion([out_c, out_c], 'bifpn')
        self.block_p4 = VoVGSCSP(out_c, out_c, n=3)

        self.fuse_p3 = Fusion([out_c, out_c], 'bifpn')
        self.block_p3 = VoVGSCSP(out_c, out_c, n=3)

        # middle attention
        self.hilo = HiLo(out_c, num_heads=8, window_size=2, alpha=0.5)

        # bottom-up
        self.down_p3 = GSConv(out_c, out_c, 3, 2)
        self.fuse_bu_p4 = Fusion([out_c, out_c], 'bifpn')
        self.block_bu_p4 = VoVGSCSP(out_c, out_c, n=3)

        self.down_p4 = GSConv(out_c, out_c, 3, 2)
        self.fuse_bu_p5 = Fusion([out_c, out_c], 'bifpn')
        self.block_bu_p5 = VoVGSCSP(out_c, out_c, n=3)

        # ASF
        self.asf = ASF([out_c, out_c, out_c], out_c)

    def forward(self, x):
        p3, p4, p5, f5 = x

        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        # top-down
        y4 = self.fuse_p4([p4, self.up(f5)])
        y4 = self.block_p4(y4)
        y4 = self.hilo(y4)

        y3 = self.fuse_p3([p3, self.up(y4)])
        y3 = self.block_p3(y3)

        # bottom-up
        o4 = self.fuse_bu_p4([y4, self.down_p3(y3)])
        o4 = self.block_bu_p4(o4)

        o5 = self.fuse_bu_p5([p5, self.down_p4(o4)])
        o5 = self.block_bu_p5(o5)

        # ASF enhancement
        asf_feat = self.asf([y3, o4, o5])
        o3 = y3 + asf_feat

        return [o3, o4, o5]


class ASF(nn.Module):
    def __init__(self, inc, channel):
        super().__init__()
        self.scalseq = ScalSeq(inc, channel)

    def forward(self, x):
        return self.scalseq(x)
