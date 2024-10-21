import torch

from prosim.dataset.data_utils import rotate
from prosim.models.utils.data import extract_agent_obs_from_center_obs

class PromptGenerator:
  def __init__(self, config):
    self.config = config
  
  def get_prompt_dim(self):
    raise NotImplementedError

  def prompt_for_scene_batch(self):
    raise NotImplementedError

  def _append_tgt_agent_info(self, result, tgt_agent_info):
    result['prompt_mask'] = tgt_agent_info['prompt_mask']
    result['agent_ids'] = tgt_agent_info['agent_ids']
    result['position'] = tgt_agent_info['position']
    result['heading'] = tgt_agent_info['heading']

    return result

  def _get_batch_tgt_agent_info(self, batch):
    # extract agent information with tgt_agent_idx
    # B: batch size
    # N: max number of agents in the batch
    # outputs:
    # goal point - (B, N, 2)
    # position - (B, N, 2)
    # heading - (B, N, 1)
    # prompt_mask - (B, N, 1)
    # agent_ids - List[List[str]]

    tgt_agent_idxs = batch.tgt_agent_idxs
    B = len(tgt_agent_idxs)
    N = max([len(tgt_idx) for tgt_idx in tgt_agent_idxs])
    device = batch.agent_hist.device

    agent_info = {}

    agent_info['goal_point'] = torch.zeros((B, N, 2), device=device)
    agent_info['position'] = torch.zeros((B, N, 2), device=device)
    agent_info['heading'] = torch.zeros((B, N, 1), device=device)
    agent_info['prompt_mask'] = torch.zeros((B, N), device=device).to(torch.bool)
    agent_info['agent_type'] = torch.zeros((B, N), device=device).to(torch.long)
    
    agent_info['agent_extend']  = torch.zeros((B, N, 2), device=device)
    agent_info['agent_vel'] = torch.zeros((B, N, 2), device=device)

    agent_info['agent_ids'] = []

    for b in range(B):
      o_idx = tgt_agent_idxs[b]
      n = len(o_idx)

      b_idx = torch.ones(n, device=device).long() * b
      t_idx = batch.agent_fut_len[b, o_idx] - 1

      if batch.agent_fut.shape[2] > 0:
        agent_info['goal_point'][b, :n] = batch.agent_fut[b_idx, o_idx, t_idx].as_format('x,y').float()
      
      agent_info['position'][b, :n] = batch.agent_hist[b_idx, o_idx, -1].as_format('x,y').float()
      agent_info['heading'][b, :n] = batch.agent_hist[b_idx, o_idx, -1].as_format('h').float()
      agent_info['prompt_mask'][b, :n] = True
      agent_info['agent_type'][b, :n] = batch.agent_type[b_idx, o_idx]
      agent_info['agent_ids'].append([batch.agent_names[b][idx] for idx in o_idx])
      agent_info['agent_extend'][b, :n] = batch.agent_hist_extent[b_idx, o_idx, -1, :2]
      agent_info['agent_vel'][b, :n] = batch.agent_hist[b_idx, o_idx, -1].as_format('xd,yd').float()

    return agent_info

  def prompt_for_batch(self, batch):
    tgt_agent_info = self._get_batch_tgt_agent_info(batch)

    prompt_dict = self.prompt_for_scene_batch(tgt_agent_info)
    
    prompt_dict['agent_type'] = tgt_agent_info['agent_type']

    if 'prompt' in prompt_dict:
      assert torch.isnan(prompt_dict['prompt'][prompt_dict['prompt_mask']]).any() == False

    return prompt_dict

  def prompt_for_rollout_batch(self, query_names, center_obs, prompt_value=None):
    agent_obs = extract_agent_obs_from_center_obs(query_names, center_obs)
    result = self._prompt_for_rollout_batch_helper(query_names, agent_obs, prompt_value)
    result['agent_type'] = agent_obs['type']

    return result


class AgentStatusGenerator(PromptGenerator):
  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.prompt_dim = 0
    
    if self.config.USE_VEL:
      self.prompt_dim += 2
    
    if self.config.USE_EXTEND:
      self.prompt_dim += 2
    
    if self.config.USE_AGENT_TYPE:
      self.prompt_dim += 3
  
  def get_prompt_dim(self):
    return self.prompt_dim

  def prompt_for_scene_batch(self, tgt_agent_info):
    '''
    # extract agent status from tgt_agent_info with tgt_agent_idx
    # B: batch size
    # N: max number of agents in the batch
    
    # outputs:
    # prompt - (B, N, D)
    # position - (B, N, 2)
    # heading - (B, N, 1)
    # prompt_mask - (B, N, 1)

    '''

    result = {}
    
    prompt = []
    if self.config.USE_VEL:
      abs_agent_vel = tgt_agent_info['agent_vel']
      agent_heading = tgt_agent_info['heading']
      agent_vel = rotate(abs_agent_vel[..., 0], abs_agent_vel[..., 1], -agent_heading[..., 0])
      prompt.append(agent_vel)
    
    if self.config.USE_EXTEND:
      prompt.append(tgt_agent_info['agent_extend'])
    
    if self.config.USE_AGENT_TYPE:
      agent_type = tgt_agent_info['agent_type']
      agent_type_one_hot = torch.zeros(agent_type.shape[0], agent_type.shape[1], 3).to(agent_type.device)
      for type_id in [1,2,3]:
        agent_type_one_hot[..., type_id-1] = (agent_type == type_id).float()
      
      prompt.append(agent_type_one_hot)
    
    prompt = torch.cat(prompt, dim=-1)
    result['prompt'] = prompt

    result = self._append_tgt_agent_info(result, tgt_agent_info)

    return result