import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np
from einops import repeat


def make_identity(N, D, device):
    if N is None:
        return torch.Tensor(np.array(np.eye(D))).to(device)
    else:
        return torch.Tensor(np.array([np.eye(D)] * N)).to(device)

def make_identity_like(A):
    assert A.shape[-2] == A.shape[-1] # Ensure A is a batch of squared matrices
    device = A.device
    shape = A.shape[:-2]
    eye = torch.eye(A.shape[-1], device=device)[(None,)*len(shape)]
    return eye.repeat(*shape, 1, 1)


def make_diagonal(vecs):
    vecs = vecs[..., None].repeat(*([1,]*len(vecs.shape)), vecs.shape[-1])
    return vecs * make_identity_like(vecs)

# Calculate Normalized Laplacian
def tracenorm_of_normalized_laplacian(A):
    D_vec = torch.sum(A, axis=-1)
    D = make_diagonal(D_vec)
    L = D - A
    inv_A_diag = make_diagonal(
        1 / torch.sqrt(1e-10 + D_vec))
    L = torch.matmul(inv_A_diag, torch.matmul(L, inv_A_diag))
    sigmas = torch.linalg.svdvals(L)
    return torch.sum(sigmas, axis=-1)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device = 'cuda'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, only_deepest=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        # import pdb; pdb.set_trace()

        for block in self.blocks:
            x = block(x)
            y = block(y)
            if not only_deepest:
                loss += torch.nn.functional.l1_loss(x, y)
        if only_deepest:
            loss = torch.nn.functional.l1_loss(x, y)
        return loss