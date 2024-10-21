import torch
from torch import nn
from prosim.models.scene_encoder.pointnet_encoder import PointNetPolylineEncoder
from prosim.models.layers.mlp import MLP
from prosim.models.layers.fourier_embedding import FourierEmbeddingFix
from prosim.dataset.motion_tag_utils import V_Action_MotionTag, V2V_MotionTag

class GoalConditionEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self.use_temp_posemd = self.config.MODEL.CONDITION_TRANSFORMER.USE_TEMPORAL_ENCODING
    self._config_model()

  def _config_model(self):
    if self.use_temp_posemd:
      self.temp_posemd = FourierEmbeddingFix(num_pos_feats=self.hidden_dim)
    self.goal_encoder = MLP([2, self.hidden_dim, self.hidden_dim], ret_before_act=True, without_norm=True)

  def forward(self, cond_input, **kwargs):
    '''
    Encode the goal condition input into goal condition embedding
    Input:
      cond_input:
        'input' (tensor): [B, C, 3] - 
          dim 0-1: relative goal position of the prompt agent's starting position
          dim - 2: the timestep at which the goal condition is valid
        'mask' (tensor): [B, C] - valid mask for the goal condition
        'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    Output:
      result['goal']:
        'emd' (tensor): [B, C, D] - the embedding of the goal condition
        'mask': pass through
        'prompt_idx': pass through
    '''
    xy_input = cond_input['input'][..., :2] # [B, C, 2]
    
    emd = self.goal_encoder(xy_input) # [B, C, D]
    if self.use_temp_posemd:
      t_input = cond_input['input'][..., 2:] # [B, C, 1]
      emd = emd + self.temp_posemd(t_input) # [B, C, D]

    result = {
      'emd': emd,
      'mask': cond_input['mask'],
      'prompt_idx': cond_input['prompt_idx'],
      'prompt_mask': cond_input['prompt_mask']
    }

    return {'goal': result}

class MotionTagEncoder(nn.Module):
  def __init__(self, config, tag_class):
    super().__init__()
    self.config = config
    all_used_tags = self.config.PROMPT.CONDITION.MOTION_TAG.USED_TAGS
    self.used_tags = [tag for tag in all_used_tags if tag in tag_class.__members__]
    self.use_temp_posemd = self.config.MODEL.CONDITION_TRANSFORMER.USE_TEMPORAL_ENCODING
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    
    self.tag_class = tag_class
    self.is_binary = self.tag_class == V2V_MotionTag
    self.param_dim = self.hidden_dim * 2 if self.is_binary else self.hidden_dim
    
    self._config_model()

  def _config_model(self):
    self.tag_encoder = nn.ParameterDict()
    for tag in self.used_tags:
      self.tag_encoder[tag] = nn.Parameter(torch.randn(self.param_dim))
    
    if self.use_temp_posemd:
      self.temp_posemd = FourierEmbeddingFix(num_pos_feats=self.hidden_dim // 2)

  def forward(self, cond_input, **kwargs):
    '''
    Encode the unary/binary motion tags and output the embedding for each tag
    Input:
      cond_input:
        'input' (tensor): [B, C, 3] - 
          dim 0: tag id of the unary tag (-1 if invalid)
          dim 1: start timestep of the action tag
          dim 2: end timestep of the action tag
        'mask' (tensor): [B, C] - valid mask for the goal condition
        'prompt_idx' (tensor): [B, C, 1/2] - index of the prompt agent in the scene (-1 if invalid)
    Output:
      dict:
        key: 
          tag_name
        value:
          'emd' (tensor): [B, T, D/2D] - the embedding of this tag_name, where T is number of tags of this tag_name
          'mask': [B, T]
          'prompt_idx': [B, T, 1]
    '''
    result = {}
    device = cond_input['input'].device
    
    B, C = cond_input['input'].shape[:2]

    B_indices = torch.arange(B, device=device)[:, None].expand(-1, C) # [B, C]
    C_indices = torch.arange(C, device=device)[None, :].expand(B, -1) # [B, C]

    p_idx_dim = 2 if self.is_binary else 1
    
    for tag in self.used_tags:
      tag_C_mask = (cond_input['input'][..., 0] == self.tag_class[tag].value)# [B, C]
      tag_sum = tag_C_mask.sum()
      
      if tag_sum == 0:
        continue
      else:
        tag_batch_cnt = tag_C_mask.sum(1) # [B]
        T = tag_batch_cnt.max() # T is the maximum number of tags across batch
        T_indices = torch.arange(T, device=device)[None, :].expand(B, -1) # [B, T]
        tag_T_mask = T_indices < tag_batch_cnt[:, None] # [B, T]

        tag_emd = torch.zeros(B, T, self.param_dim, device=device) # [B, T, D/2D]
        tag_valid_mask = torch.zeros(B, T, device=device, dtype=torch.bool) # [B, T]
        tag_prompt_idx = torch.ones(B, T, p_idx_dim, device=device, dtype=torch.long) * -1 # [B, T, 1/2]

        tag_bidx = B_indices[tag_C_mask] # [V] - batch indices of the valid tags
        tag_cidx = C_indices[tag_C_mask] # [V] - column indices of the valid tags
        tag_valid_mask[tag_T_mask] = cond_input['mask'][tag_bidx, tag_cidx] # [V]
        tag_prompt_idx[tag_T_mask] = cond_input['prompt_idx'][tag_bidx, tag_cidx] # [V, 1/2]

        V = tag_C_mask.sum()
        tag_emd_valid = self.tag_encoder[tag][None, :].expand(V, -1) # [V, D/2D]
        if self.use_temp_posemd:
          tag_time_input = cond_input['input'][tag_bidx, tag_cidx, 1:3] # [V, 2]
          tag_time_emd = self.temp_posemd(tag_time_input) # [V, D]
          if self.is_binary:
            tag_time_emd = tag_time_emd.repeat(1, 2) # [V, 2D]
          tag_emd_valid = tag_emd_valid + tag_time_emd # [V, D/2D]
        
        tag_emd[tag_T_mask] = tag_emd_valid # [B, T, D/2D]

      tag_dict = {'emd': tag_emd, 'mask': tag_valid_mask, 'prompt_idx': tag_prompt_idx}
      result[tag] = tag_dict
  
    return result


class V_ActionTagEncoder(MotionTagEncoder):
  def __init__(self, config):
    super().__init__(config, V_Action_MotionTag)

class V2V_MotionTagEncoder(MotionTagEncoder):
  def __init__(self, config):
    super().__init__(config, V2V_MotionTag)

class DragPointEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self.pointnet_cfg = self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.DRAG_POINTS

    self._config_model()

  def _config_model(self):
    self.pointnet_encoder = PointNetPolylineEncoder(2, self.hidden_dim, self.pointnet_cfg)

  def forward(self, cond_input, **kwargs):
    '''
    Encode the drag point condition input into condition embedding
    Input:
      cond_input:
        'input' (tensor): [B, C, T, 2] - 
          dim 0-1: relative drag point position of the prompt agent's starting position
        'mask' (tensor): [B, C] - valid mask for the goal condition
        'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    Output:
      result['drag_point']:
        'emd' (tensor): [B, C, D] - the embedding of the drag points condition
        'mask': pass through
        'prompt_idx': pass through
    '''
    drag_points = cond_input['input'] # [B, C, T, 2]
    drag_points_mask = ~(drag_points.isnan().any(-1)) # [B, C, T]

    emd = self.pointnet_encoder(drag_points, drag_points_mask) # [B, C, D]

    result = {
      'emd': emd,
      'mask': cond_input['mask'],
      'prompt_idx': cond_input['prompt_idx'],
      'prompt_mask': cond_input['prompt_mask']
    }

    return {'drag_point': result}


condition_encoders = {'goal': GoalConditionEncoder, 'v_action_tag': V_ActionTagEncoder, 'drag_point': DragPointEncoder, 'v2v_tag': V2V_MotionTagEncoder}