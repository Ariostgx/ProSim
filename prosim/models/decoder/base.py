import torch
import torch.nn as nn

from prosim.models.layers.mlp import MLP
from prosim.core.registry import registry

@registry.register_decoder(name='base')
class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_dim = config.HIDDEN_DIM
    self.goal_cfg = config.DECODER.GOAL_PRED
    self.K = self.goal_cfg.K
    self._config_models()
  
  def _config_models(self):
    if self.goal_cfg.ENABLE:
      self.goal_prob_head = MLP([self.hidden_dim, self.hidden_dim//2, self.K], ret_before_act=True)
      self.goal_point_head = MLP([self.hidden_dim, self.hidden_dim//2, self.K * 2], ret_before_act=True)

  def _goal_pred(self, goal_input, prompt_enc, result):
    '''
    Input:
      goal_input: [B, N, D]
    Output:
      result['goal_prob']: [B, N, K]
      result['goal_point']: [B, N, K, 2]
    '''

    B, N, D = goal_input.shape
    
    K = self.K
    result_goal_prob = torch.zeros(B, N, K).to(goal_input.device)
    result_goal_point = torch.zeros(B, N, K, 2).to(goal_input.device)
        
    prompt_mask = prompt_enc['prompt_mask']
    
    # [Q, D]
    valid_k_input = goal_input[prompt_mask]

    # [Q, K]
    goal_prob = self.goal_prob_head(valid_k_input).view(-1, K)

    # [Q, K, 2]
    goal_point = self.goal_point_head(valid_k_input).view(-1, K, 2)

    result_goal_prob[prompt_mask] = goal_prob
    result_goal_point[prompt_mask] = goal_point

    # [B, N, K]
    result['goal_prob'] = result_goal_prob

    # [B, N, K, 2]
    result['goal_point'] = result_goal_point

    return result


  def _fusion(self, scene_emb, prompt_emd, prompt_mask):
    raise NotImplementedError

  def forward(self, scene_emb, prompt_enc):
    result = {}

    prompt_emd = prompt_enc['prompt_emd']
    prompt_mask = prompt_enc['prompt_mask']

    result['emd'] = self._fusion(scene_emb, prompt_emd, prompt_mask)

    if self.goal_cfg.ENABLE:
      self._goal_pred(scene_emb, result['emd'], result)
    
    return result