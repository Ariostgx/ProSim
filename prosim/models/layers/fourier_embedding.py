# Mostly copied from https://github.com/ZikangZhou/QCNet/blob/main/layers/fourier_embedding.py
import math
from typing import List, Optional

import torch
import torch.nn as nn

from prosim.models.utils.weight_init import weight_init


class FourierEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
                for _ in range(input_dim)])
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self,
                continuous_inputs: Optional[torch.Tensor] = None,
                categorical_embs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError('Both continuous_inputs and categorical_embs are None')
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)

class FourierEmbeddingFix(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000) -> None:
        super(FourierEmbeddingFix, self).__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, continuous_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = continuous_inputs

        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        D = pos.shape[-1]
        pos_dims = []
        for i in range(D):
            pos_dim_i = pos[..., i, None] / dim_t
            pos_dim_i = torch.stack((pos_dim_i[..., 0::2].sin(), pos_dim_i[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_dims.append(pos_dim_i)
        posemb = torch.cat(pos_dims, dim=-1)
        
        return posemb