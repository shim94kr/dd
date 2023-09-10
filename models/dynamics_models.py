import torch
import torch.nn as nn
import numpy as np
from utils.loss_utils import make_identity_like, tracenorm_of_normalized_laplacian, make_identity, make_diagonal
import einops

def _rep_M(M, T):
    return einops.repeat(M, "n a1 a2 -> n t a1 a2", t=T)

def _loss(A, B):
    return torch.sum((A-B)**2)

def _solve(A, B):
    ATA = A.transpose(-2, -1) @ A
    ATB = A.transpose(-2, -1) @ B
    return torch.linalg.solve(ATA, ATB)

def loss_bd(M_star, alignment):
    # Block Diagonalization Loss
    S = torch.abs(M_star)
    STS = torch.matmul(S.transpose(-2, -1), S)
    if alignment:
        laploss_sts = tracenorm_of_normalized_laplacian(
            torch.mean(STS, 0))
    else:
        laploss_sts = torch.mean(
            tracenorm_of_normalized_laplacian(STS), 0)
    return laploss_sts

def loss_orth(M_star):
    # Orthogonalization of M
    I = make_identity_like(M_star)
    return torch.mean(torch.sum((I-M_star @ M_star.transpose(-2, -1))**2, axis=(-2, -1)))


class LinearTensorDynamicsLSTSQ(nn.Module):
    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.permute = torch.randperm(M.shape[0])
            self.M = M[self.permute]

        def __call__(self, H):
            H0, H1 = H[:, 0], H[:, 1]
            return H0 @ self.M

    def __init__(self, alignment=True):
        super().__init__()
        self.alignment = alignment

    def __call__(self, H):
        # Regress M.
        # H0.shape = H1.shape [Batch, 2, m, a]
        H0, H1 = H[:, 0], H[:, 1]
        # Batch x dim_m x dim_a
        # The difference between the the time shifted components
        loss_internal_0 = _loss(H0, H1)
        M_star = _solve(H0, H1)
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(H0 @ M_star, H1)
        losses = (loss_bd(dyn_fn.M, self.alignment),
                    loss_orth(dyn_fn.M), loss_internal_T)
        
        # M_star is returned in the form of module, not the matrix
        return dyn_fn