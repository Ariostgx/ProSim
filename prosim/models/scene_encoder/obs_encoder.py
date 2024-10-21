import torch.nn as nn
from prosim.models.layers.mlp import MLP
from .pointnet_encoder import PointNetPolylineEncoder

def get_obs_input_dim(cfg):
  in_dim = len(cfg.DATASET.FORMAT.HISTORY.ELEMENTS.split(','))
  
  if cfg.DATASET.FORMAT.HISTORY.WITH_EXTEND:
    in_dim += 2
  
  if cfg.DATASET.FORMAT.HISTORY.WITH_AGENT_TYPE:
    in_dim += 3
  
  if cfg.DATASET.FORMAT.HISTORY.WITH_TIME_EMB:
    in_dim += cfg.DATASET.FORMAT.HISTORY.STEPS
  
  return in_dim

class MLP_OBV_ENCODER(nn.Module):
  def __init__(self, cfg, model_cfg):
    self.cfg = cfg
    self.hidden_dim = cfg.MODEL.HIDDEN_DIM
    self.pool_func = cfg.MODEL.OBS_ENCODER.MLP.POOL
    super().__init__()
    self._config_enc()

  def _config_enc(self):
    hist_dim = get_obs_input_dim(self.cfg)

    if self.pool_func == 'none':
      hist_step = self.cfg.DATASET.FORMAT.HISTORY.STEPS
      input_dim = hist_step * hist_dim
    else:
      input_dim = hist_dim

    self.hist_encoder = MLP([input_dim, self.hidden_dim // 2, self.hidden_dim], ret_before_act=True)

  def _pool_hist(self, hist_enc, hist_mask):
    # hist_enc: [B, N, T, D]
    # hist_mask: [B, N, T]

    if self.pool_func == 'mean':
      hist_enc = hist_enc.masked_fill(~hist_mask[..., None], 0.0)
      hist_enc = hist_enc.sum(dim=2) / hist_mask.sum(dim=2, keepdim=True)
      hist_enc = hist_enc.masked_fill(~hist_mask.any(dim=2, keepdim=True), 0.0)
    
    elif self.pool_func == 'max':
      hist_enc = hist_enc.masked_fill(~hist_mask[..., None], -1e9)
      hist_enc = hist_enc.max(dim=2)[0]
    
    else:
      raise NotImplementedError
    
    return hist_enc

  def forward(self, batch_obs):
    B, N = batch_obs['input'].shape[:2]
    obs_mask = batch_obs['mask'].all(dim=-1) # [B, N, T]

    # avoid propagation of nan values
    obs_input = batch_obs['input'].masked_fill(~obs_mask[..., None], 0.0) # [B, N, T, d]

    if self.pool_func == 'none':
      obs_input = obs_input.reshape(B, N, -1) # [B, N, T*d]
      obs_mask = obs_mask.all(dim=-1) # [B, N]
      hist_enc = self.hist_encoder(obs_input) # [B, N, D]
    
    else:
      hist_enc = self.hist_encoder(obs_input) # [B, N, T, D]
      hist_enc = self._pool_hist(hist_enc, obs_mask) # [B, N, D]
      obs_mask = obs_mask.any(dim=-1) # [B, N]
    
    return hist_enc, obs_mask

class POINTNET_OBV_ENCODER(PointNetPolylineEncoder):
  def __init__(self, cfg, model_cfg):
    in_dim = get_obs_input_dim(cfg)
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    layer_cfg = cfg.MODEL.OBS_ENCODER.POINTNET
    super().__init__(in_dim, hidden_dim, layer_cfg)
  
  def forward(self, batch_obs):
    obs_input = batch_obs['input']
    obs_mask = batch_obs['mask'].all(dim=-1) # [B, N, T]

    hist_enc = super().forward(obs_input, obs_mask)
    return hist_enc, obs_mask.any(dim=-1) # [B, N]


obs_encoders = {'mlp': MLP_OBV_ENCODER, 'pointnet': POINTNET_OBV_ENCODER}