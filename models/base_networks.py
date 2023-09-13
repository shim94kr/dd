import numpy as np
import torch
from torch import nn
from models.resblock import Block, Conv1d1x1Block
from einops.layers.torch import Rearrange
from einops import repeat

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
                 num_classes,
                 k=1,
                 act=nn.ReLU(),
                 kernel_size=3,
                 n_blocks=3):
        super().__init__()
        
        ch_base = int(32 * k)
        chs = [ch_base * (2 ** i) for i in range(n_blocks + 1)]

        self.phi = nn.Sequential(
            nn.LazyConv2d(chs[0], 3, 1, 1),
            *[Block(chs[i], chs[i+1], chs[i+1],
                    resample='down', activation=act, kernel_size=kernel_size) for i in range(n_blocks)],
            nn.GroupNorm(min(32, chs[n_blocks]), chs[n_blocks]),
            act)

        self.linear_cls = nn.LazyLinear(num_classes)

    def embed (self, x):
        h = x
        h = self.phi(h)
        h = h.mean(dim=[-1, -2])
        return h

    def get_logits(self, x):
        return self.linear_cls(x)

    def __call__(self, x):
        h = x
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear_cls(h)
        return h


class ResNetDecoder(nn.Module):
    def __init__(self, ch_x, k=1, act=nn.ReLU(), kernel_size=3, bottom_width=4, n_blocks=3):
        super().__init__()
        ch_base = int(32 * k)
        chs = [ch_base * (2 ** i) for i in range(n_blocks + 1)]

        self.bottom_width = bottom_width
        self.linear = nn.LazyLinear(chs[n_blocks])
        self.net = nn.ModuleList(nn.Sequential(
            Block(chs[i+1], chs[i], chs[i],
                    resample='up', activation=act, kernel_size=kernel_size)) for i in range(n_blocks-1, -1, -1)
        )

        self.net_last = nn.Sequential(
            nn.GroupNorm(min(32, chs[0]), chs[0]),
            act,
            nn.Conv2d(chs[0], ch_x, 3, 1, 1)
        )

    def __call__(self, x):
        x = self.linear(x)
        x = repeat(x, 'n c -> n c h w',
                   h=self.bottom_width, w=self.bottom_width)

        for i in range(len(self.net)):
            x = self.net[i](x)

        x = self.net_last(x)
        return x
