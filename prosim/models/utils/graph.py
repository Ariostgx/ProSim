# Copied from https://github.com/ZikangZhou/QCNet/blob/main/utils/graph.py

from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import coalesce
from torch_geometric.utils import degree


def add_edges(
        from_edge_index: torch.Tensor,
        to_edge_index: torch.Tensor,
        from_edge_attr: Optional[torch.Tensor] = None,
        to_edge_attr: Optional[torch.Tensor] = None,
        replace: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from_edge_index = from_edge_index.to(device=to_edge_index.device, dtype=to_edge_index.dtype)
    mask = ((to_edge_index[0].unsqueeze(-1) == from_edge_index[0].unsqueeze(0)) &
            (to_edge_index[1].unsqueeze(-1) == from_edge_index[1].unsqueeze(0)))
    if replace:
        to_mask = mask.any(dim=1)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr[~to_mask], from_edge_attr], dim=0)
        to_edge_index = torch.cat([to_edge_index[:, ~to_mask], from_edge_index], dim=1)
    else:
        from_mask = mask.any(dim=0)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr, from_edge_attr[~from_mask]], dim=0)
        to_edge_index = torch.cat([to_edge_index, from_edge_index[:, ~from_mask]], dim=1)
    return to_edge_index, to_edge_attr


def merge_edges(
        edge_indices: List[torch.Tensor],
        edge_attrs: Optional[List[torch.Tensor]] = None,
        reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)
