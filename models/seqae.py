import torch
import torch.nn as nn
from einops import rearrange
from models.base_networks import ResNetDecoder, Conv1d1x1Encoder
from models.dynamics_models import LinearTensorDynamicsLSTSQ


class SeqAELSTSQ(nn.Module):
    def __init__(self, config, encoders, **kwargs):
        super().__init__()
        self.dim_a = config.model.dim_a
        self.dim_m = config.model.dim_m
        self.predictive = config.model.predictive
        self.ch_x = config.dataset.num_channel
        self.k = config.model.k
        self.kernel_size = 3
        self.n_blocks = 3
        self.bottom_width = 4

        self.encs = encoders
        self.dec = ResNetDecoder(self.ch_x, k=self.k, kernel_size=self.kernel_size, bottom_width=self.bottom_width, n_blocks=self.n_blocks)
        self.linears = nn.ModuleList(nn.LazyLinear(self.dim_a * self.dim_m) for i in range(len(self.encs)))
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=config.model.alignment)
        if config.model.change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def encode(self, xs, rand_idx):
        shape = xs.shape
        xs = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))

        enc = self.encs[rand_idx]
        linear = self.linears[rand_idx]

        H = enc.embed(xs)
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))

        logits = enc.get_logits(H)
        
        return H, logits

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        H = rearrange(H, 'n t d_s -> (n t) d_s')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def __call__(self, xs, rand_idx):
        # Encoded Latent. Batch x 2 x  dim_m x dim_a
        H, logits = self.encode(xs, rand_idx)

        # ==Esitmate dynamics==
        permute_logits = torch.randperm(H.shape[0])
        permute_H = torch.randperm(H.shape[0])
        logits_diff = logits[:, 0] - logits[:, 1]
        H_diff = H[:, 0] - H[:, 1]
        #H_diff = H_diff.roll(shifts=(1,), dims=(0,))
        H_preds = torch.stack([H[:, 0], H[:,0] + H_diff[permute_H], H[:, 1]], dim=1)
        #M_star = self.dynamics_model(H)
        #permute = torch.randperm(M_star.shape[0])
        #M_diff = M_star - torch.eye(self.dim_a).reshape(1, self.dim_a, self.dim_a).to(M_star.device)
        #H_preds = torch.stack([H[:, 0], H[:, 1] @ M_star[permute], H[:, 1]], dim=1)

        # Prediction in the observation space
        x_preds = self.decode(H_preds)
        x_diff = xs[:, 0:1] - x_preds[:, 0:2]
        logits = torch.stack([logits[:,0], logits[:,0] + logits_diff[permute_logits]])
        return x_preds, x_diff, H_diff, logits