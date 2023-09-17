import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from models.base_networks import ResNetDecoder, Conv1d1x1Encoder
from models.dynamics_models import LinearTensorDynamicsLSTSQ


class SeqAELSTSQ(nn.Module):
    def __init__(self, config, encoders, **kwargs):
        super().__init__()
        self.feat_dim = config.model.feat_dim
        self.predictive = config.model.predictive
        self.ch_x = config.dataset.num_channel
        self.num_classes = config.dataset.num_classes
        self.k = config.model.k
        self.kernel_size = 3
        self.n_blocks = 3
        self.bottom_width = 4

        self.encs = encoders
        self.dec = ResNetDecoder(self.ch_x, k=self.k, kernel_size=self.kernel_size, bottom_width=self.bottom_width, n_blocks=self.n_blocks)

        self.linears = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(enc.last_dim, self.feat_dim)
            ) if enc.last_dim != self.feat_dim else nn.Identity() for enc in self.encs
        )

        self.linear_feat = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim)
        )

        self.linear_cls = nn.Linear(self.feat_dim, 10)
        
    def encode(self, x, w, rand_idx):
        enc = self.encs[rand_idx]
        linear = self.linears[rand_idx]
        
        H = linear(enc.embed(x))
        
        logits = torch.einsum('bh,ch->bc', H, w.detach()) / np.sqrt(H.shape[1])

        return  H, logits

    def decode(self, Hs):

        n, t = Hs.shape[:2]
        Hs = rearrange(Hs, 'n t d_s -> (n t) d_s')
        xs = self.dec(Hs)
        xs = torch.reshape(xs, (n, t, *xs.shape[1:]))
        return xs