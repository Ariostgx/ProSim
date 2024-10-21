# Copied from: https://github.com/sshaoshuai/MTR/blob/master/mtr/models/utils/polyline_encoder.py
# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 


import torch
import torch.nn as nn
from prosim.models.layers.mlp import MLP

        # # in_channels, hidden_dim, num_layers=3, num_pre_layers=1

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_cfg):
        super().__init__()

        num_layers = layer_cfg.NUM_MLP_LAYERS
        num_pre_layers = layer_cfg.NUM_PRE_LAYERS
        
        self.pre_mlps = MLP([in_dim] + [hidden_dim] * num_pre_layers, ret_before_act=False)
        self.mlps = MLP([hidden_dim * 2] + [hidden_dim] * (num_layers - num_pre_layers), ret_before_act=False)
        self.out_mlps = MLP([hidden_dim] * 3, without_norm=True, ret_before_act=True)
        
    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # print('polylines.dtype:', polylines.dtype)
        # print('polylines_mask.dtype:', polylines_mask.dtype)

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        model_dtype = polylines_feature_valid.dtype
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1], dtype=model_dtype)
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1], dtype=model_dtype)
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        valid_mask = (polylines_mask.sum(dim=-1) > 0)
        feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
        feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1], dtype=model_dtype)
        feature_buffers[valid_mask] = feature_buffers_valid

        # print('feature_buffers.dtype:', feature_buffers.dtype)

        return feature_buffers