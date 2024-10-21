import torch
import math

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Copied from https://github.com/OpenDriveLab/UniAD/blob/main/projects/mmdet3d_plugin/models/utils/functional.py
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def pos2posemb(pos, num_pos_feats=128, temperature=10000):

    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    D = pos.shape[-1]
    posembs = []
    for i in range(D):
        pos_i = pos[..., i, None] / dim_t
        pos_i = torch.stack((pos_i[..., 0::2].sin(), pos_i[..., 1::2].cos()), dim=-1).flatten(-2)
        posembs.append(pos_i)
    
    posemb = torch.cat(posembs, dim=-1)
    
    return posemb