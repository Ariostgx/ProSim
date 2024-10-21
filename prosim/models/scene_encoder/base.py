import torch
import torch.nn as nn
import torch.nn.functional as F


from prosim.core.registry import registry
from .map_encoder import map_encoders
from .obs_encoder import obs_encoders

@registry.register_scene_encoder(name='base')
class SceneEncoder(nn.Module):
  def __init__(self, config, model_cfg):
    super().__init__()
    self.config = config
    self.model_cfg = model_cfg
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self._config_models()

  def _config_models(self):    
    self.map_encoder = map_encoders[self.model_cfg.MAP_TYPE](self.config, self.model_cfg)
    self.obs_encoder = obs_encoders[self.model_cfg.OBS_TYPE](self.config, self.model_cfg)

    self._config_fusion()
  
  def _config_fusion(self):
    raise NotImplementedError
  
  def _scene_fusion(self, map_emd, map_mask, obs_emd, obs_mask):
    raise NotImplementedError

  def forward(self, batch_obs, batch_map):
    # inputs:
    # batch_obs encode the observation of the agent (other agents): {'input', 'mask'}
    # batch_map encode the map information of the agent (other agents) {'input', 'mask'}

    # output dict:
    # scene_tokens: the tokens of the scene elements
    # scene_mask: the mask of the scene elements
    # scene_emd: a D-dim emb for each scene in the batch

    map_emd, map_mask = self.map_encoder(batch_map)
    obs_emd, obs_mask = self.obs_encoder(batch_obs)

    result = self._scene_fusion(batch_map, batch_obs, map_emd, obs_emd, map_mask, obs_mask)

    return result
