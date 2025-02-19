from typing import List

import torch
from torch import nn

def z_norm(x, dims: List[int], eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value

def _glob_norm(x, eps: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)

class _LayerNorm(nn.Module):
    ''' Layer Normalization base class. '''
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)

class GlobLN(_LayerNorm):
    ''' Global Layer Normalization (globLN) .'''
    def forward(self, x, EPS: float = 1e-8):
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)
