import random
import torch
from torch.nn.utils.rnn import pad_sequence

from prosim.dataset.data_utils import rotate
from prosim.models.utils.data import  extract_agent_obs_from_center_obs

class PromptGenerator:
  def __init__(self, config):
    self.config = config
  
  def get_prompt_dim(self):
    raise NotImplementedError

  def prompt_for_agent(self, agent_dict):
    raise NotImplementedError

  def prompt_for_scene_batch(self, batch, tgt_agent_info, split=None):
    raise NotImplementedError

  def _sample_prompt(self, result, SCENE_MAX_AGENT, rand_sample):
    B = len(result['agent_ids'])
    agent_nums = [len(names) for names in result['agent_ids']]
    sampled_result = {key: [] for key in result.keys()}

    for b in range(B):
      N = agent_nums[b]
      agent_idx = torch.where(result['prompt_mask'][b])[0].tolist()
      assert len(agent_idx) == N

      # if not 'rand_sample', then use all the agents
      if rand_sample and N > SCENE_MAX_AGENT:
        sample_idx = random.sample(agent_idx, SCENE_MAX_AGENT)
      else:
        sample_idx = agent_idx

      for key in result.keys():
        if key == 'agent_ids':
          sampled_result[key].append([result[key][b][idx] for idx in sample_idx])
        else:
          sampled_result[key].append(result[key][b, sample_idx])

    for key in sampled_result.keys():
      if key == 'agent_ids':
        continue

      sampled_result[key] = pad_sequence(sampled_result[key], batch_first=True, padding_value=0.0)

    return sampled_result

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
      n_idx = tgt_agent_idxs[b]
      n = len(n_idx)

      b_idx = torch.ones(n, device=device).long() * b
      t_idx = batch.agent_fut_len[b, n_idx] - 1

      if batch.agent_fut.shape[2] > 0:
        agent_info['goal_point'][b, :n] = batch.agent_fut[b_idx, n_idx, t_idx].as_format('x,y').float()
      
      agent_info['position'][b, :n] = batch.agent_hist[b_idx, n_idx, -1].as_format('x,y').float()
      agent_info['heading'][b, :n] = batch.agent_hist[b_idx, n_idx, -1].as_format('h').float()
      agent_info['prompt_mask'][b, :n] = True
      agent_info['agent_type'][b, :n] = batch.agent_type[b_idx, n_idx]
      agent_info['agent_ids'].append([batch.agent_names[b][idx] for idx in n_idx])
      agent_info['agent_extend'][b, :n] = batch.agent_hist_extent[b_idx, n_idx, -1, :2]
      agent_info['agent_vel'][b, :n] = batch.agent_hist[b_idx, n_idx, -1].as_format('xd,yd').float()

    return agent_info

  def prompt_for_batch(self, batch, mode='scene', rand_sample=False, SCENE_MAX_AGENT=None, split='train'):
    tgt_agent_info = self._get_batch_tgt_agent_info(batch)

    if mode == 'scene':
      prompt_dict = self.prompt_for_scene_batch(batch, tgt_agent_info, split)
    else:
      raise NotImplementedError
    
    prompt_dict['agent_type'] = tgt_agent_info['agent_type']

    # we do sampling for the tgt agent index instead of here 
    
    # prompt_dict = self._sample_prompt(prompt_dict, SCENE_MAX_AGENT, rand_sample)

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

  def prompt_for_scene_batch(self, batch, tgt_agent_info, split=None):
    
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


prompt_generators = {'agent_status': AgentStatusGenerator}
