import torch
import torch.nn as nn
from einops import rearrange
from models.base_networks import ResNetDecoder, Conv1d1x1Encoder
from models.dynamics_models import LinearTensorDynamicsLSTSQ


class SeqAELSTSQ(nn.Module):
    def __init__(self, config, encoders, **kwargs):
        super().__init__()
        self.dim_m = config.model.dim_m
        self.dim_a = config.model.dim_a
        self.predictive = config.model.predictive
        self.ch_x = config.dataset.num_channel
        self.num_classes = config.dataset.num_classes
        self.k = config.model.k
        self.kernel_size = 3
        self.n_blocks = 3
        self.bottom_width = 4

        self.encs = encoders
        self.dec = ResNetDecoder(self.ch_x, k=self.k, kernel_size=self.kernel_size, bottom_width=self.bottom_width, n_blocks=self.n_blocks)

        last_dim = self.encs[0].last_dim
        self.linears = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(last_dim, last_dim)
            )
            for enc in self.encs
        )
        self.linear_feat = nn.Linear(last_dim, last_dim * self.bottom_width * self.bottom_width)
        self.linear_cls = nn.Linear(last_dim, self.num_classes)
        
        
    def encode(self, x, rand_idx):
        enc = self.encs[rand_idx]
        linear = self.linears[rand_idx]
        
        feat, H = enc.embed(x)
        logits = self.linear_cls(H)

        '''
        H = linear(H)
        logits = self.linear_cls(H)

        H = H.reshape(H.shape[0], self.dim_m, self.dim_a)
        return H, logits
        '''
        return feat, H, logits

    def decode(self, Hs, feat_real):

        n, t = Hs.shape[:2]
        xs = self.dec(Hs, feat_real)
        xs = torch.reshape(xs, (n, t+1, *xs.shape[1:]))
        return xs