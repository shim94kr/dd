import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.ops_utils import WeightStandarization, WeightStandarization1d
import torch.nn.utils.parametrize as P

from einops.layers.torch import Rearrange
from einops import repeat
from einops import rearrange

class Conv1d1x1Encoder(nn.Sequential):
    def __init__(self,
                 dim_out=16,
                 dim_hidden=128,
                 act=nn.ReLU()):
        super().__init__(
            nn.LazyConv1d(dim_hidden, 1, 1, 0),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Rearrange('n c s -> n s c'),
            nn.LayerNorm((dim_hidden)),
            Rearrange('n s c-> n c s'),
            act,
            nn.LazyConv1d(dim_out, 1, 1, 0)
        )


class ResNetEncoder(nn.Module):
    def __init__(self,
                 num_channel,
                 num_classes,
                 k=1,
                 act=nn.ReLU(),
                 kernel_size=3,
                 n_blocks=3):
        super().__init__()
        
        ch_base = int(32 * k)
        chs = [ch_base * (2 ** i) for i in range(n_blocks + 1)]

        self.init_conv = nn.Conv2d(num_channel, chs[0], 3, 1, 1)
        self.phi = nn.ModuleList(
            [Block(chs[i], chs[i+1], chs[i+1], resample='down', activation=act, kernel_size=kernel_size) for i in range(n_blocks)]
        ) 
       
        self.last_layer = nn.Sequential(
            nn.GroupNorm(min(32, chs[n_blocks]), chs[n_blocks]),
            act
        )
        
        self.last_dim = chs[n_blocks] * 4 * 4

    def embed (self, x):
        h = self.init_conv(x)
        for phi in self.phi:
            h = phi(h)
        
        h = self.last_layer(h)
        #h = h.mean(dim=[-1, -2])
        return h.flatten(1,3)


class ResNetDecoder(nn.Module):
    def __init__(self, ch_in, k=1, act=nn.ReLU(), kernel_size=3, bottom_width=4, n_blocks=3):
        super().__init__()
        ch_base = int(32 * k)
        chs = [ch_base * (2 ** i) for i in range(n_blocks + 1)]

        self.bottom_width = bottom_width
        self.linear = None
        if ch_in != chs[-1] * bottom_width * bottom_width:
            self.linear = nn.Linear(ch_in, chs[-1] * bottom_width * bottom_width)

        self.net = nn.ModuleList(nn.Sequential(
            Block(chs[i+1], chs[i], chs[i],
                    resample='up', activation=act, kernel_size=kernel_size)) for i in range(n_blocks-1, -1, -1)
        )

        self.net_last = nn.Sequential(
            nn.GroupNorm(min(32, chs[0]), chs[0]),
            act,
            nn.Conv2d(chs[0], 3, 3, 1, 1)
        )

    def __call__(self, x):
        if self.linear is not None:
            x = self.linear(x)
        x = x.reshape(x.shape[0], -1, self.bottom_width, self.bottom_width)
        for i in range(len(self.net)):
            x = self.net[i](x)

        x = self.net_last(x)
        return x

def upsample_conv(x, conv):
    # Upsample -> Conv
    x = nn.Upsample(scale_factor=2, mode='nearest')(x)
    x = conv(x)
    return x


def conv_downsample(x, conv):
    # Conv -> Downsample
    x = conv(x)
    h = F.avg_pool2d(x, 2)
    return h


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 kernel_size=3,
                 padding=None,
                 activation=F.relu,
                 resample=None,
                 group_norm=True,
                 skip_connection=True):
        super(Block, self).__init__()
        if padding is None:
            padding = (kernel_size-1) // 2

        self.skip_connection = skip_connection
        self.activation = activation
        self.resample = resample
        if self.resample is None or self.resample == 'up':
            hidden_channels = out_channels if hidden_channels is None else hidden_channels
        else:
            hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels,
                            kernel_size=kernel_size, padding=padding)
        self.c2 = nn.Conv2d(hidden_channels, out_channels,
                            kernel_size=kernel_size, padding=padding)

        initializer = torch.nn.init.xavier_uniform_
        initializer(self.c1.weight, math.sqrt(2))
        initializer(self.c2.weight, math.sqrt(2))
        P.register_parametrization(
            self.c1, 'weight', WeightStandarization())
        P.register_parametrization(
            self.c2, 'weight', WeightStandarization())

        if group_norm:
            self.b1 = nn.GroupNorm(min(32, in_channels), in_channels)
            self.b2 = nn.GroupNorm(min(32, hidden_channels), hidden_channels)
        else:
            self.b1 = self.b2 = lambda x: x

        if self.skip_connection:
            self.c_sc = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, padding=0)
            initializer(self.c_sc.weight)

    def residual(self, x):
        x = self.b1(x)
        x = self.activation(x)
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.resample == 'down':
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        # Upsample -> Conv
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = self.c_sc(x)

        elif self.resample == 'down':
            x = self.c_sc(x)
            x = F.avg_pool2d(x, 2)
        else:
            x = self.c_sc(x)
        return x

    def __call__(self, x):
        if self.skip_connection:
            return self.residual(x) + self.shortcut(x)
        else:
            return self.residual(x)


class Conv1d1x1Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 act=F.relu):
        super().__init__()

        self.act = act
        initializer = torch.nn.init.xavier_uniform_
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)
        self.c2 = nn.Conv1d(hidden_channels, out_channels, 1, 1, 0)
        initializer(self.c1.weight, math.sqrt(2))
        initializer(self.c2.weight, math.sqrt(2))
        P.register_parametrization(
            self.c1, 'weight', WeightStandarization1d())
        P.register_parametrization(
            self.c2, 'weight', WeightStandarization1d())
        self.norm1 = nn.LayerNorm((in_channels))
        self.norm2 = nn.LayerNorm((hidden_channels))
        self.c_sc = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        initializer(self.c_sc.weight)

    def residual(self, x):
        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.c1(x)
        x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.c2(x)
        return x

    def shortcut(self, x):
        x = self.c_sc(x)
        return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)