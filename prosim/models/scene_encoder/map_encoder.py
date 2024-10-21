import torch.nn as nn
from prosim.models.layers.mlp import MLP
from .pointnet_encoder import PointNetPolylineEncoder

class MLP_MAP_ENCODER(nn.Module):
  def __init__(self, cfg, model_cfg):
    super().__init__()
    self.config = cfg
    self.model_cfg = model_cfg
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self.pool_func = self.config.MODEL.MAP_ENCODER.MLP.POOL
    
    self._config_models()

  def _config_models(self):
    self.lane_encode = MLP([4, 256, 512, self.hidden_dim], ret_before_act=True)
    
    self.type_embedding = nn.Embedding(4, self.hidden_dim)
    self.traf_embedding = nn.Embedding(4, self.hidden_dim)

  def map_lane_encode(self, lane_inp):
    polyline = lane_inp[..., :4]
    polyline_type = lane_inp[..., 4].to(int)
    polyline_traf = lane_inp[..., 5].to(int) + 1

    polyline_type_embed = self.type_embedding(polyline_type)
    polyline_traf_embed = self.traf_embedding(polyline_traf)

    lane_enc = self.lane_encode(polyline) + polyline_traf_embed + polyline_type_embed

    return lane_enc

  def _pool_hist(self, lane_enc, lane_mask):
    # lane_enc: [B, M, N, D]

    if self.pool_func == 'mean':
      lane_enc = lane_enc.masked_fill(~lane_mask[..., None], 0.0)
      lane_enc = lane_enc.sum(dim=2) / lane_mask.sum(dim=2, keepdim=True)
      lane_enc = lane_enc.masked_fill(~lane_mask.any(dim=2, keepdim=True), 0.0)
    
    elif self.pool_func == 'max':
      lane_enc = lane_enc.masked_fill(~lane_mask[..., None], -1e9)
      lane_enc = lane_enc.max(dim=2)[0]
    
    else:
      raise NotImplementedError
    
    return lane_enc


  def forward(self, batch_map):
    # encode the map into scene embedding
    map_input = batch_map['input']
    map_mask = batch_map['mask'].any(dim=-1) # [B, M, P]

    lane_enc = self.map_lane_encode(map_input)

    # lane_enc: [B, M, N, D]
    # M: number of lanes
    # N: the number of points for each lane
    if len(lane_enc.shape) == 4:
      lane_mask = map_input[..., 4] > 0 # [B, M, N]
      lane_enc = self._pool_hist(lane_enc, lane_mask) # [B, M, D]
    
    return lane_enc, map_mask

class POINTNET_MAP_ENCODER(PointNetPolylineEncoder):
  def __init__(self, cfg, model_cfg):
    in_dim = 6
    
    if cfg.DATASET.FORMAT.MAP.WITH_TYPE_EMB:
      in_dim += 3

    if cfg.DATASET.FORMAT.MAP.WITH_DIR:
      in_dim += 2
      
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    layer_cfg = cfg.MODEL.MAP_ENCODER.POINTNET
    super().__init__(in_dim, hidden_dim, layer_cfg)
  
  def forward(self, batch_map):
    # encode the map into scene embedding
    map_input = batch_map['input']
    map_mask = batch_map['mask'] # [B, M, P]

    lane_enc = super().forward(map_input, map_mask)

    return lane_enc, map_mask.any(dim=-1) # [B, M]

map_encoders = {'mlp': MLP_MAP_ENCODER, 'pointnet': POINTNET_MAP_ENCODER}