import os
import json
import torch
import random
import glob
import re
import numpy as np
import tqdm

from prosim.dataset.motion_tag_utils import V_Action_MotionTag, V2V_MotionTag
from prosim.dataset.text_utils import AGENT_TEMPLATE

import multiprocessing
from multiprocessing import Pool

def load_text_file(txt_file):
    snapshot_id = txt_file.split('/')[-1].split('.')[0]
    
    if 'waymo' in txt_file:
      scene_id = '_'.join(snapshot_id.split('_')[:2])
    else:
      scene_id = snapshot_id.split('_')[0]
    
    with open(txt_file, 'r') as f:
        raw_lines = f.readlines()
        cleaned_lines = process_lines(raw_lines)  # Assume process_lines is defined elsewhere
    return scene_id, snapshot_id, cleaned_lines


def process_lines(lines):
  # Remove the line number and extra space
  lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines]

  # Remove the first introduction line with "Here are"
  lines = [re.sub(r'Here are.*', '', line) for line in lines]

  # Remove empty lines and extra quotes
  lines = [line for line in lines if line]
  lines = [line.replace('"', '') for line in lines]

  return lines


def load_all_llm_texts(llm_input_folder):
  print('Condition: Loading all LLM text files from folder: ', llm_input_folder)
  llm_txt_files = glob.glob(f'{llm_input_folder}/*.txt')
  llm_data = {}

  for txt_file in tqdm.tqdm(llm_txt_files, desc='Loading LLM text files'):
      snapshot_id = txt_file.split('/')[-1].split('.')[0]
      scene_id = snapshot_id.split('_')[0]

      if scene_id not in llm_data:
          llm_data[scene_id] = {}

      with open(txt_file, 'r') as f:
          raw_lines = f.readlines()

          # Process the lines by cleaning and filtering them
          cleaned_lines = process_lines(raw_lines)

          llm_data[scene_id][snapshot_id] = cleaned_lines

  print('Condition: Loaded LLM text data for', len(llm_data), 'scenes.')
  return llm_data


def get_closest_llm_text(scene_id, scene_start_ts, llm_data, max_time_shift=40):
  """
  Retrieves the closest snapshot text data for a given scene ID and start timestamp,
  considering a maximum allowed time shift.
  
  Args:
  scene_id (str): The ID of the scene to process.
  scene_start_ts (int): The start timestamp of the scene.
  llm_data (dict): Dictionary containing LLM data structured by scene and snapshot IDs.
  max_time_shift (int): Maximum permissible time difference in timestamps.
  
  Returns:
  list: Text data from the closest snapshot within the permissible time shift.
        Returns an empty list if no snapshot is close enough or if scene_id is not found.
  """
  if scene_id in llm_data:
      snapshot_ids = list(llm_data[scene_id].keys())
      snapshot_ts = [int(snapshot_id.split('_')[-3]) for snapshot_id in snapshot_ids]

      # print(snapshot_ids)
      
      # Calculate absolute differences from the scene timestamp
      snapshot_ts_diff = np.abs(np.array(snapshot_ts) - scene_start_ts)
      closest_snapshot_idx = np.argmin(snapshot_ts_diff)
      
      # Check if the closest snapshot is within the acceptable time shift
      if snapshot_ts_diff[closest_snapshot_idx] > max_time_shift:
          return []
      else:
          closest_snapshot_id = snapshot_ids[closest_snapshot_idx]
          return llm_data[scene_id][closest_snapshot_id]
  else:
      print(f"Scene ID {scene_id} not found in LLM data.")
      return []

class BatchCondition:
  def __init__(self, all_cond: dict):
    self.all_cond = all_cond
  
  def __to__(self, device, non_blocking=False):
    for cond_type in self.all_cond.keys():
        for cond_key in self.all_cond[cond_type].keys():
            if type(self.all_cond[cond_type][cond_key]) in [list, str]:
                continue
            self.all_cond[cond_type][cond_key] = self.all_cond[cond_type][cond_key].to(device, non_blocking=non_blocking)
    
    return self

  def __getitem__(self, key):
    assert key in self.all_cond.keys()
    return self.all_cond[key]

  def __len__(self):
    return len(self.all_cond)
  
  def keys(self):
    return self.all_cond.keys()

def get_goal_condition_batch(batch, cond_cfg):
  '''
  Obtain all the goal conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, 3] - 
        dim 0-1: relative goal position of the prompt agent's starting position
        dim - 2: the timestep at which the goal condition is valid
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    
    In this function, C = N, where N is the number of prompt agents in the batch
  '''
  tgt_agent_idxs = batch.tgt_agent_idxs
  B = len(tgt_agent_idxs)
  N = max([len(tgt_idx) for tgt_idx in tgt_agent_idxs])
  device = batch.agent_hist.device

  goal_point = torch.zeros((B, N, 2), device=device)
  goal_fut_len = torch.zeros((B, N, 1), device=device)
  valid_mask = torch.zeros((B, N), device=device).to(torch.bool)
  prompt_idx = torch.ones((B, N, 1), device=device).to(torch.long) * -1

  io_pair_batch = batch.extras['io_pairs_batch']

  for b in range(B):
    o_idx = tgt_agent_idxs[b]
    n = len(o_idx)

    b_idx = torch.ones(n, device=device).long() * b
    n_idx = torch.arange(n, device=device)

    goal_point[b, :n] = io_pair_batch['goal'][b_idx, 0, n_idx]
    valid_mask[b, :n] = True
    prompt_idx[b, :n] = n_idx.view(-1, 1)
    goal_fut_len[b, :n] = batch.agent_fut_len[b, o_idx].float().view(-1, 1)

    goal_agent_names = io_pair_batch['agent_names'][b]
    prompt_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][b]

    assert all([goal_agent_names[i] == prompt_agent_names[i] for i in range(n)]), f"Goal Condition agent names: {goal_agent_names}, Prompt agent names: {prompt_agent_names}"
  
  # First two dimensions are the relative goal position of the prompt agent's starting position
  # Third dimension is the timestep at which the goal condition is valid (float value) - this is to help the model figure out when the goal condition should be reached
  input = torch.cat([goal_point, goal_fut_len], dim=-1)
  
  return {'input': input, 'mask': valid_mask, 'prompt_idx': prompt_idx}

def get_v_action_tag_condition_batch(batch, cond_cfg):
  '''
  Obtain all the unary action tag conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, 3] - 
        dim 0: the unique action tag of the prompt agent
        dim 1: start timestep of the action tag
        dim 2: end timestep of the action tag
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene

    C is the maximum number of condition tags in the batch
  '''
  tgt_agent_idxs = batch.tgt_agent_idxs
  B = len(tgt_agent_idxs)
  device = batch.agent_hist.device

  all_input_list = []
  all_prompt_idx_list = []

  for b in range(B):
    p_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][b]

    # Get the action tags ids for the prompt agents
    batch_inputs = []
    batch_prompt_idx = []
    for tag in batch.extras['motion_tag'][b]:
       if tag['type'] == 'unary' and tag['agents'][0] in p_agent_names:
          tag_id = V_Action_MotionTag[tag['tag']].value
          start_t, end_t = tag['interval']
          batch_inputs.append([tag_id, start_t, end_t])
          batch_prompt_idx.append(p_agent_names.index(tag['agents'][0]))
    
    all_input_list.append(torch.tensor(batch_inputs, device=device, dtype=torch.long).view(-1, 3))
    all_prompt_idx_list.append(torch.tensor(batch_prompt_idx, device=device, dtype=torch.long).view(-1, 1))

  # Pad the sequences
  input = torch.nn.utils.rnn.pad_sequence(all_input_list, batch_first=True, padding_value=-1)   # [B, C, 3]
  prompt_idx = torch.nn.utils.rnn.pad_sequence(all_prompt_idx_list, batch_first=True, padding_value=-1)   # [B, C, 1]
  valid_mask = (input != -1).all(dim=-1)   # [B, C]
    
  return {'input': input, 'mask': valid_mask, 'prompt_idx': prompt_idx}

def get_llm_text_condition_batch(batch, cond_cfg, llm_texts):
  '''
  Obtain all the goal conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    llm_text_condition: dict
      'input' (List[str]): [B, C] - list of goal condition texts, empty string if no llm text condition
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, A] - index of the prompt agent in the scene
    
    A is the maximum number of prompt agents of each text in the batch
  '''
  tgt_agent_idxs = batch.tgt_agent_idxs
  B = len(tgt_agent_idxs)
  device = batch.agent_hist.device

  all_input_list = []
  all_prompt_idx_list = []

  for b in range(B):
    p_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][b]
    p_agent_names_short = [name[:5] for name in p_agent_names]

    # Get the llm texts for each batch
    batch_inputs = []
    batch_prompt_idx = []

    if llm_texts is None or len(llm_texts) == 0:
      scene_llm_data = batch.extras['llm_texts'][b]
    else:
      scene_llm_data = get_closest_llm_text(batch.scene_ids[b], batch.scene_ts[b].item(), llm_texts)

    for text in scene_llm_data:
      names = re.findall(r'<([a-zA-Z0-9]+)>', text)

      # print(names)

      text_pidxs = []
      for name in names:
        name = name.lower()
        if name in p_agent_names_short:
          pidx = p_agent_names_short.index(name)
          text = text.replace(f'<{name}>', AGENT_TEMPLATE.format(pidx))
          text_pidxs.append(pidx)
      
      # pass if all the mentioned agents are not in the prompt agents
      if len(text_pidxs) == 0 and len(names) > 0:
        continue

      if cond_cfg.OneText.USE_PLACEHOLDER:
        if len(text_pidxs) > 0:
          text = AGENT_TEMPLATE.format(text_pidxs[0]) + ' is there.'
        else:
          text = 'placeholder.'

      batch_inputs.append(text)
      batch_prompt_idx.append(torch.tensor(text_pidxs, device=device, dtype=torch.long))
    
    if len(batch_prompt_idx) > 0:
      batch_prompt_idx = torch.nn.utils.rnn.pad_sequence(batch_prompt_idx, batch_first=True, padding_value=-1)   # [C, A]
    
    all_input_list.append(batch_inputs)
    all_prompt_idx_list.append(batch_prompt_idx)

  # Pad the sequences
  C = max([len(inputs) for inputs in all_input_list])
  for inputs in all_input_list:
    inputs += [''] * (C - len(inputs))
  
  acnts = []
  for idx in all_prompt_idx_list:
    if len(idx) > 0:
      acnts.append(idx.shape[1])
    else:
      acnts.append(0)
  A = max(acnts)
  for b in range(B):
    idx = all_prompt_idx_list[b]
    if len(idx) == 0:
      all_prompt_idx_list[b] =  torch.ones((0, A), device=device, dtype=torch.long) * -1
    else:
      all_prompt_idx_list[b] = torch.cat([idx, torch.ones((idx.shape[0], A - idx.shape[1]), device=device, dtype=torch.long) * -1], dim=-1)

  prompt_idx = torch.nn.utils.rnn.pad_sequence(all_prompt_idx_list, batch_first=True, padding_value=-1)   # [B, C, A]
  
  valid_mask = [[len(text)>0 for text in texts] for texts in all_input_list]   # [B, C]
  valid_mask = torch.tensor(valid_mask, device=device, dtype=torch.bool)
  
  return {'input': all_input_list, 'mask': valid_mask, 'prompt_idx': prompt_idx}


def get_v2v_tag_condition_batch(batch, cond_cfg):
  '''
  Obtain all the binary action tag conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, 3] - 
        dim 0: the unique action tag of the prompt agent
        dim 1: start timestep of the action tag
        dim 2: end timestep of the action tag
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 2] - index of the prompt agent in the scene - 
        dim 0: index of the first agent in the binary action tag
        dim 1: index of the second agent in the binary action tag
    C is the maximum number of condition tags in the batch
  '''
  tgt_agent_idxs = batch.tgt_agent_idxs
  B = len(tgt_agent_idxs)
  device = batch.agent_hist.device

  all_input_list = []
  all_prompt_idx_list = []

  for b in range(B):
    p_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][b]

    # Get the action tags ids for the prompt agents
    batch_inputs = []
    batch_prompt_idx = []
    for tag in batch.extras['motion_tag'][b]:
       if tag['type'] == 'binary':
        if tag['agents'][0] in p_agent_names and tag['agents'][1] in p_agent_names:
          tag_id = V2V_MotionTag[tag['tag']].value
          start_t, end_t = tag['interval']
          batch_inputs.append([tag_id, start_t, end_t])
          batch_prompt_idx.append([p_agent_names.index(agent) for agent in tag['agents']])
    
    all_input_list.append(torch.tensor(batch_inputs, device=device, dtype=torch.long).view(-1, 3))
    all_prompt_idx_list.append(torch.tensor(batch_prompt_idx, device=device, dtype=torch.long).view(-1, 2))

  # Pad the sequences
  input = torch.nn.utils.rnn.pad_sequence(all_input_list, batch_first=True, padding_value=-1)   # [B, C, 3]
  prompt_idx = torch.nn.utils.rnn.pad_sequence(all_prompt_idx_list, batch_first=True, padding_value=-1)   # [B, C, 2]
  valid_mask = (input != -1).all(dim=-1)   # [B, C]
    
  return {'input': input, 'mask': valid_mask, 'prompt_idx': prompt_idx}

def random_subset_mask(valid_mask, min_len=1, max_len=None):
  """
  Generates a random subset mask within the valid durations of the mask.

  Args:
  valid_mask (torch.Tensor): A boolean mask of shape [B, N, T] indicating valid timesteps.
  min_len (int): Minimum length of the random subset.
  max_len (int): Maximum length of the random subset. If None, uses the length of the valid duration.

  Returns:
  torch.Tensor: A new mask of shape [B, N, T] with random subsets selected within valid durations.
  """
  B, N, T = valid_mask.shape
  new_mask = torch.zeros_like(valid_mask)

  for b in range(B):
      for n in range(N):
          valid_indices = valid_mask[b, n].nonzero(as_tuple=True)[0]
          if len(valid_indices) > 0:
              valid_start = valid_indices[0]
              valid_end = valid_indices[-1]

              valid_len = min(max_len, valid_end - valid_start + 1)

              if valid_len > min_len:
                subset_len = torch.randint(min_len, valid_len, (1,))
              else:
                subset_len = valid_len

              subset_start = torch.randint(valid_start, valid_end - subset_len + 2, (1,))

              new_mask[b, n, subset_start:subset_start+subset_len] = 1

  return new_mask

def get_drag_points_condition_batch(batch, cond_cfg):
  '''
  Obtain all the mouse drag conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, T, 2] - 
        T: number of points of the drag action
        dim 0-1: relative goal position of the prompt agent's starting position
      'mask' (tensor): [B, C] - valid mask for the drag condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
  
  Mouse Drag Points Randomization:
    - Random subset: randomly select a consecutive subset of points from the drag action
    - Raddom noise: add random noise to the points of the drag action
  '''
  drag_cfg = cond_cfg.DRAG_POINTS
  point_sample_rate = drag_cfg.POINT_SAMPLE_RATE

  device = batch.agent_hist.device
  io_pair_batch = batch.extras['io_pairs_batch']

  full_traj_xy = io_pair_batch['full_traj_xy']
  
  B, N, T_full, _ = full_traj_xy.shape
  T = T_full // point_sample_rate

  drag_points = full_traj_xy[:, :, ::point_sample_rate, :].clone() # [B, N, T, 2]
  point_valid_mask = ~(drag_points.isnan().all(dim=-1)) # [B, N, T]
  prompt_idx = torch.arange(N, device=device).view(1, -1, 1).expand(B, -1, -1).clone() # [B, N, 1]

  if drag_cfg.RANDOM_SUBSET:
    # uniformly sample a consecutive subset of points from the drag action
    point_subsample_mask = random_subset_mask(point_valid_mask, min_len=drag_cfg.MIN_SUBSET_LEN, max_len=T) # [B, N, T]
    point_valid_mask &= point_subsample_mask # [B, N, T]
    drag_points[~point_valid_mask] *= torch.nan # [B, N, T, 2]
  
  if drag_cfg.RANDOM_NOISE > 0:
    noise = torch.randn_like(drag_points) * drag_cfg.RANDOM_NOISE
    drag_points += noise
  
  valid_mask = point_valid_mask.any(dim=-1) # [B, N]
  prompt_idx[~valid_mask] = -1

  return {'input': drag_points, 'mask': valid_mask, 'prompt_idx': prompt_idx}

def get_motion_tag_text_condition_batch(batch, cond_cfg, split, template_dict):
  '''
  Obtain all templated motion tags for the prompt agents in the batch (currently only supports unary motion tags)
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
    split: str - 'train' / 'val' / 'test' / 'rollout'
    template_dict: dict - template for the motion tags
  Output:
    goal_condition: dict
      'input' (List[str]): [B, C] - list of motion tag templated texts; empty string if no motion tag
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    C is the maximum number of condition tags in the batch
  '''
  tgt_agent_idxs = batch.tgt_agent_idxs
  B = len(tgt_agent_idxs)
  device = batch.agent_hist.device

  all_input_list = []
  all_prompt_idx_list = []

  for b in range(B):
    p_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][b]

    # Get the action tags ids for the prompt agents
    batch_inputs = []
    batch_prompt_idx = []
    for tag in batch.extras['motion_tag'][b]:
       if tag['type'] == 'unary' and tag['agents'][0] in p_agent_names:
          tag_name = tag['tag']
          start_t, end_t = tag['interval']
          
          # Get the template for the motion tag
          tidx_s, tidx_e = cond_cfg.MOTION_TAG_TEXT.REPHRASE_IDX[split.upper()]
          templates = template_dict[tag_name][tidx_s:tidx_e]
          template = random.choice(templates)
          
          if cond_cfg.MOTION_TAG_TEXT.USE_AGENT_NAME:
            agent_name = tag['agents'][0][:5]
          else:
            agent_name = 'agent'
          
          # input = template.format(agent_name=agent_name, start_t=start_t, end_t=end_t)
          input = template.format(agent_name=agent_name)
          batch_inputs.append(input)

          batch_prompt_idx.append(p_agent_names.index(tag['agents'][0]))
    
    all_input_list.append(batch_inputs)
    all_prompt_idx_list.append(torch.tensor(batch_prompt_idx, device=device, dtype=torch.long).view(-1, 1))

  # Pad the sequences
  C = max([len(inputs) for inputs in all_input_list])
  for inputs in all_input_list:
    inputs += [''] * (C - len(inputs))

  prompt_idx = torch.nn.utils.rnn.pad_sequence(all_prompt_idx_list, batch_first=True, padding_value=-1)   # [B, C, 1]
  valid_mask = prompt_idx[..., 0] != -1   # [B, C]
    
  return {'input': all_input_list, 'mask': valid_mask, 'prompt_idx': prompt_idx}

def get_goal_text_condition_batch(batch, cond_cfg):
  '''
  Obtain all the goal conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_text_condition: dict
      'input' (List[str]): [B, C] - list of goal condition texts, empty string if no goal condition
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    
    In this function, C = N, where N is the number of prompt agents in the batch
  '''
  goal_point_condition = get_goal_condition_batch(batch, cond_cfg)
  goal_text_condition = {'input': [], 'mask': goal_point_condition['mask'], 'prompt_idx': goal_point_condition['prompt_idx']}

  for bidx in range(len(goal_point_condition['input'])):
    b_texts = []
    for cidx in range(len(goal_point_condition['input'][bidx])):
      if goal_point_condition['mask'][bidx, cidx]:
        goal_pos = goal_point_condition['input'][bidx, cidx, :2].cpu().tolist()
        # goal_text = f"agent goal point ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) at timestep {goal_point_condition['input'][bidx, cidx, 2]:.0f}"
        goal_text = f"agent goal point ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})"
      else:
        goal_text = ""
      b_texts.append(goal_text)
    goal_text_condition['input'].append(b_texts)
  
  return goal_text_condition

condition_funcs = {'goal': get_goal_condition_batch, 'v_action_tag': get_v_action_tag_condition_batch, 'drag_point': get_drag_points_condition_batch, 'v2v_tag': get_v2v_tag_condition_batch, 'motion_tag_text': get_motion_tag_text_condition_batch, 'motion_tag_OneText': get_motion_tag_text_condition_batch, 'goal_text': get_goal_text_condition_batch, 'goal_OneText': get_goal_text_condition_batch, 'llm_text': get_llm_text_condition_batch, 'llm_text_OneText': get_llm_text_condition_batch}

def get_goal_condition_caption(batch, goal_cond):
  return "shown as green cross"

def get_v_action_tag_caption(batch, v_tag_cond):
  B, C, _ = v_tag_cond['input'].shape
  tag_captions = []

  batch_agent_names = batch.extras['io_pairs_batch']['agent_names']

  bidx = 0 # only consider the first batch for visualization purposes
  for cidx in range(C):
    if v_tag_cond['mask'][bidx, cidx] == 0:
      continue
    
    tag_idx = v_tag_cond['input'][bidx, cidx, 0].cpu().item()
    tag_name = V_Action_MotionTag(tag_idx).name

    start_t, end_t = v_tag_cond['input'][bidx, cidx, 1:].cpu().tolist()
    
    agent_name = batch_agent_names[bidx][v_tag_cond['prompt_idx'][bidx][cidx]][:5]
    
    tag_captions.append(f"{tag_name}({agent_name}: {start_t}-{end_t})")
  
  v_tag_caption = ', '.join(tag_captions)
  return v_tag_caption

def get_drag_point_caption(batch, cond):
  return "shown as blue dots"


def get_v2v_tag_caption(batch, v_tag_cond):
  B, C, _ = v_tag_cond['input'].shape
  tag_captions = []

  batch_agent_names = batch.extras['io_pairs_batch']['agent_names']

  bidx = 0 # only consider the first batch for visualization purposes
  for cidx in range(C):
    if v_tag_cond['mask'][bidx, cidx] == 0:
      continue
    
    tag_idx = v_tag_cond['input'][bidx, cidx, 0].cpu().item()
    tag_name = V2V_MotionTag(tag_idx).name

    start_t, end_t = v_tag_cond['input'][bidx, cidx, 1:].cpu().tolist()
    
    agent_name_1 = batch_agent_names[bidx][v_tag_cond['prompt_idx'][bidx][cidx][0]][:5]
    agent_name_2 = batch_agent_names[bidx][v_tag_cond['prompt_idx'][bidx][cidx][1]][:5]
    
    tag_captions.append(f"{tag_name}({agent_name_1}, {agent_name_2}: {start_t}-{end_t})")
  
  v_tag_caption = ', '.join(tag_captions)
  return v_tag_caption

def get_motion_tag_text_caption(batch, motion_tag_cond):
  B, C, _ = motion_tag_cond['prompt_idx'].shape
  tag_captions = []

  batch_agent_names = batch.extras['io_pairs_batch']['agent_names']

  bidx = 0 # only consider the first batch for visualization purposes
  for cidx in range(C):
    if motion_tag_cond['mask'][bidx, cidx] == False:
      continue
    
    tag_text = motion_tag_cond['input'][bidx][cidx]
    agent_name = batch_agent_names[bidx][motion_tag_cond['prompt_idx'][bidx][cidx][0]][:5]
    
    tag_captions.append(f"{tag_text}({agent_name})")
  
  motion_tag_caption = '\n'.join(tag_captions)
  return motion_tag_caption

def get_OneText_caption(batch, motion_tag_cond):
  bidx = 0

  if motion_tag_cond['mask'][bidx] == False:
    return ""
  
  return motion_tag_cond['input'][bidx]

def get_goal_text_caption(batch, goal_text_cond):
  B, C, _ = goal_text_cond['prompt_idx'].shape
  goal_captions = []

  bidx = 0 # only consider the first batch for visualization purposes
  for cidx in range(C):
    if goal_text_cond['mask'][bidx, cidx] == False:
      continue
    
    goal_text = goal_text_cond['input'][bidx][cidx]
    goal_captions.append(goal_text)
  
  goal_caption = '\n'.join(goal_captions)
  return goal_caption

caption_funcs = {'goal': get_goal_condition_caption, 'v_action_tag': get_v_action_tag_caption, 'drag_point': get_drag_point_caption, 'v2v_tag': get_v2v_tag_caption, 'motion_tag_text': get_motion_tag_text_caption, 'motion_tag_OneText': get_OneText_caption, 'goal_text': get_goal_text_caption, 'goal_OneText': get_OneText_caption, 'llm_text': get_goal_text_caption, 'llm_text_OneText': get_OneText_caption}

def sample_cond_data(data, sample_method='uniform', sample_rate=0.5, random_shuffle=True, sample_rate_std=0.2, max_cond_per_batch=None, max_cond_per_scene=None):
  '''
  Sample and form data based on various sampling methods and conditions. 
  Supports handling both tensor and list of strings as input data.

  Input:
    data: dict containing 'input', 'mask', and 'prompt_idx'
      - 'input': Can be a tensor [B, C, ...] or a list of strings. Represents the data to be sampled.
      - 'mask': Tensor [B, C], where each element is a boolean indicating whether the condition at that index is valid.
      - 'prompt_idx': Tensor [B, C, ...], indexes of the prompt agent in the scene.
    sample_method: string (default 'uniform')
      - 'fix_sample_rate': Samples a fixed percentage (defined by sample_rate) of the valid conditions.
      - 'uniform': Randomly samples a varying number of valid conditions in each batch.
      - 'normal_sample_rate': Samples based on a rate drawn from a normal distribution centered around sample_rate.
    sample_rate: float (default 0.5), used with 'fix_sample_rate' and 'normal_sample_rate' to determine the sampling rate.
    random_shuffle: boolean (default True), if True, samples are randomly selected, otherwise the first N valid samples are selected.
    max_cond_per_batch: int (default None), maximum number of conditions to sample per batch.
    max_cond_per_scene: int (default None), maximum number of conditions to sample per scene.

  Output:
    new_data: dict containing 'input', 'mask', and 'prompt_idx'
      - 'input': A tensor or a list of strings, depending on the input type. If tensor, size is [B, sampled_C, ...] where sampled_C is the number of samples after applying the mask and sampling method.
      - 'mask': Tensor [B, sampled_C], updated mask after sampling.
      - 'prompt_idx': Tensor [B, sampled_C, ...], updated prompt indices after sampling.
  '''
  # Unpack the data
  input_data = data['input']
  mask_tensor = data['mask']
  prompt_idx_tensor = data['prompt_idx']

  B, C = len(input_data), len(mask_tensor[0])

  # Initialize lists to store new data
  new_input_list = []
  new_mask_list = []
  new_prompt_idx_list = []

  # Check if input is a list of strings
  input_is_string = isinstance(input_data[0], list)

  batch_quota = max_cond_per_batch if max_cond_per_batch is not None else B * C

  # Iterate over each batch
  for bidx in range(B):
      # Find indices where the condition is valid (mask is true)
      valid_indices = [i for i, valid in enumerate(mask_tensor[bidx]) if valid]

      # Determine the number of samples based on the sample method
      if sample_method == 'fix_sample_rate':
          num_samples = int(len(valid_indices) * sample_rate)
      elif sample_method == 'uniform':
          num_samples = random.randint(0, len(valid_indices))
      elif sample_method == 'normal_sample_rate':
          rate = min(1, max(0, random.normalvariate(sample_rate, sample_rate_std)))
          num_samples = int(len(valid_indices) * rate)
      elif sample_method == 'none':
          num_samples = len(valid_indices)
      else:
          raise ValueError("Invalid sampling method")
    
      num_samples = min(num_samples, max_cond_per_scene) if max_cond_per_scene is not None else num_samples
      num_samples = min(num_samples, batch_quota)
      batch_quota -= num_samples

      # Randomly sample or take the first N samples based on random_shuffle
      if num_samples > 0:
          if random_shuffle:
              sample_nidx = random.sample(valid_indices, num_samples)
          else:
              sample_nidx = valid_indices[:num_samples]
      else:
          sample_nidx = []

      # Select the samples from each batch
      if input_is_string:
          new_input_list.append([input_data[bidx][i] for i in sample_nidx] if sample_nidx else [])
      else:
          new_input_list.append(input_data[bidx, sample_nidx] if sample_nidx else torch.empty(0, *input_data.shape[2:]))

      new_mask_list.append(mask_tensor[bidx, sample_nidx] if sample_nidx else torch.empty(0, dtype=mask_tensor.dtype))
      new_prompt_idx_list.append(prompt_idx_tensor[bidx, sample_nidx] if sample_nidx else torch.empty(0, *prompt_idx_tensor.shape[2:]))

  # Handle the list of strings case
  if input_is_string:
      # Padding is not needed for a list of strings
      new_input = new_input_list
  else:
      # Pad the sequences for tensors
      new_input = torch.nn.utils.rnn.pad_sequence(new_input_list, batch_first=True, padding_value=-1.0)

  new_mask = torch.nn.utils.rnn.pad_sequence(new_mask_list, batch_first=True, padding_value=False)
  new_prompt_idx = torch.nn.utils.rnn.pad_sequence(new_prompt_idx_list, batch_first=True, padding_value=-1).long()

  batch_C = new_mask.sum(dim=1)
  print('DEBUG: number of condition samples in each batch:', batch_C.tolist())

  # Form new data
  new_data = {
      'input': new_input,
      'mask': new_mask,
      'prompt_idx': new_prompt_idx
  }

  return new_data

def concat_text_to_OneText(batch, text_data, cond_cfg, cond_type):
  '''
  Concatenate the motion tag text for each batch into a single string.

  Input:
    text_data: dict containing 'input', 'mask', and 'prompt_idx'
      - 'input': List[str]: [B, C] - list of motion tag templated texts; empty string if no motion tag
      - 'mask': [B, C] tensor - valid mask for the goal condition
      - 'prompt_idx': [B, C] tensor - index of the prompt agent in the scene
      - 'prompt_mask': [B, N] tensor - mask for the prompt agents in the scene
  
  Output:
    OneText_data: dict containing 'input', 'mask', and 'prompt_idx'
      - 'input': List[str]: [B] - list of concatenated motion tag texts for each batch
      - 'mask': [B] tensor - valid mask for the goal condition
      - 'prompt_mask': [B, N] tensor - mask for the prompt agents in the scene
  '''

  concat_texts = []

  for bidx in range(len(text_data['input'])):
    b_texts = []
    for cidx in range(len(text_data['input'][bidx])):
      if text_data['mask'][bidx, cidx]:
        tag_text = text_data['input'][bidx][cidx]
        if 'agent' in tag_text and cond_type != 'llm_text_OneText':
          pidx = text_data['prompt_idx'][bidx, cidx].item()
          if cond_cfg.OneText.USE_PLACEHOLDER:
            tag_text = AGENT_TEMPLATE.format(pidx) + ' is there.'
          else:
            tag_text = tag_text.replace('agent', AGENT_TEMPLATE.format(pidx))
    
        b_texts.append(tag_text)
    
    if cond_cfg.OneText.SHUFFLE_TEXT:
      random.shuffle(b_texts)

    concat_texts.append('\n'.join(b_texts))
  
  mask = text_data['mask'].any(dim=-1)
  # prompt_mask = batch.extras['prompt']['motion_pred']['prompt_mask'] & mask[:, None]

  prompt_mask = text_data['prompt_mask']

  return {'input': concat_texts, 'mask': mask, 'prompt_mask': prompt_mask}

def goal_condition_batch_from_option(agent_cond_dict, cond_cfg, device):
  '''
  Obtain all the goal conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, 3] - 
        dim 0-1: relative goal position of the prompt agent's starting position
        dim - 2: the timestep at which the goal condition is valid
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene
    
    B = 1 for this function
  '''
  goal_xyt_list = []
  prompt_idx_list = []
  for key, value in agent_cond_dict.items():
    if value['Enable Goal Condition']:
      goal_xyt_list.append([value['Goal - X'], value['Goal - Y'], 80.0])
      prompt_idx_list.append(int(value['Agent ID'].split('A')[1]))
  
  input_data = torch.tensor(goal_xyt_list).to(device)[None, ] # [1, C, 3]
  mask = torch.ones((1, input_data.shape[1]), device=device, dtype=torch.bool) # [1, C]
  prompt_idx = torch.tensor(prompt_idx_list)[None, ..., None].to(device) # [1, 1, 1]

  return {'input': input_data, 'mask': mask, 'prompt_idx': prompt_idx}

def v_action_tag_condition_batch_from_option(agent_cond_dict, cond_cfg, device):
  '''
  Obtain all the unary action tag conditions for the prompt agents in the batch
  Input:
    batch: SceneBatch
    cond_cfg: ConditionConfig
  Output:
    goal_condition: dict
      'input' (tensor): [B, C, 3] - 
        dim 0: the unique action tag of the prompt agent
        dim 1: start timestep of the action tag
        dim 2: end timestep of the action tag
      'mask' (tensor): [B, C] - valid mask for the goal condition
      'prompt_idx' (tensor): [B, C, 1] - index of the prompt agent in the scene

    C is the maximum number of condition tags in the batch
  '''

  tag_t_list = []
  prompt_idx_list = []

  for key, value in agent_cond_dict.items():
    nidx = int(value['Agent ID'].split('A')[1])
    
    for tag_type in ['Lane Tag', 'Turn Tag', 'Speed Tag']:
      if value[tag_type] == 'Null':
        continue

      tag_id = V_Action_MotionTag[value[tag_type]].value
      tag_t_list.append([tag_id, 0.0, 80.0])
      prompt_idx_list.append(nidx)

  input_data = torch.tensor(tag_t_list).to(device)[None, ] # [1, C, 3]
  mask = torch.ones((1, input_data.shape[1]), device=device, dtype=torch.bool) # [1, C]
  prompt_idx = torch.tensor(prompt_idx_list)[None, ..., None].to(device) # [1, 1, 1]
  
  return {'input': input_data, 'mask': mask, 'prompt_idx': prompt_idx}

cond_from_option_funcs = {'goal': goal_condition_batch_from_option, 'v_action_tag': v_action_tag_condition_batch_from_option}


def _mask_priority_condition(all_cond, priority_order):
  """
  Masks conditions based on a priority order to ensure that each agent has only one type of condition applied,
  even when conditions can apply to multiple agents.

  Args:
  all_cond (dict): Dictionary containing all conditions data by type.
  priority_order (list): List of condition types in order of priority.

  Returns:
  dict: Updated all_cond with masks adjusted to enforce priority.
  """
  # Determine the maximum number of agents by checking the largest agent index across all conditions
  max_agents = 0
  for cond_type, data in all_cond.items():
    if data['prompt_idx'].nelement() > 0:
      max_agents = max(max_agents, data['prompt_idx'].max().item() + 1)  # +1 because indices start from 0

  # Initialize a tensor to track the highest priority condition applied to each agent
  agent_priority = torch.full((max_agents,), len(priority_order), dtype=torch.int)  # Set to max length as default no condition assigned

  # Iterate over conditions by priority to determine which condition each agent should have
  for priority, cond_type in enumerate(priority_order):
    if cond_type not in all_cond:
      continue

    condition_data = all_cond[cond_type]
    masks = condition_data['mask']
    prompt_idxs = condition_data['prompt_idx']

    for b in range(masks.size(0)):  # Loop over batches
        for c in range(masks.size(1)):  # Loop over conditions in the batch
            if masks[b, c]:  # Check if the condition is initially valid
                agent_indices = prompt_idxs[b, c]  # Get the agent indices for this condition
                for agent_idx in agent_indices:
                    if agent_idx < max_agents and priority < agent_priority[agent_idx]:
                        agent_priority[agent_idx] = priority

  # Now adjust masks based on the determined highest priority conditions for each agent
  for cond_type in all_cond:
    condition_data = all_cond[cond_type]
    masks = condition_data['mask']
    prompt_idxs = condition_data['prompt_idx']

    current_priority = priority_order.index(cond_type) if cond_type in priority_order else len(priority_order)

    for b in range(masks.size(0)):
        for c in range(masks.size(1)):
            agent_indices = prompt_idxs[b, c]
            # Disable the mask if this condition's priority is not the highest for any involved agent
            if not all(agent_priority[agent_idx] == current_priority for agent_idx in agent_indices if agent_idx < max_agents):
                masks[b, c] = False
                prompt_idxs[b, c] = agent_indices * 0 - 1

  return all_cond

def _mask_soft_priority_condition(all_cond, priority_scores):
  """
  Masks conditions based on a soft priority strategy to ensure that each agent has conditions applied
  based on priority scores, allowing for some randomness.

  Args:
  all_cond (dict): Dictionary containing all conditions data by type.
  priority_scores (dict): Dictionary with priority scores for each condition type.

  Returns:
  dict: Updated all_cond with soft priority masks adjusted.
  """
  # Determine the maximum number of agents by checking the largest agent index across all conditions
  max_agents = 0
  for cond_type, data in all_cond.items():
      if data['prompt_idx'].nelement() > 0:
          max_agents = max(max_agents, data['prompt_idx'].max().item() + 1)  # +1 because indices start from 0

  # Collect all condition types and their indices for each agent
  agent_conditions = {i: [] for i in range(max_agents)}
  for cond_type, data in all_cond.items():
      masks = data['mask']
      prompt_idxs = data['prompt_idx']

      for b in range(masks.size(0)):  # Loop over batches
          for c in range(masks.size(1)):  # Loop over conditions in the batch
              if masks[b, c]:  # Check if the condition is initially valid
                  agents = prompt_idxs[b, c]
                  for agent in agents:
                      agent = agent.item()
                      if agent != -1:
                        agent_conditions[agent].append((cond_type, b, c))

  # Apply a soft priority selection based on priority scores
  for agent, conditions in agent_conditions.items():
      if len(conditions) > 1:
          # Get probabilities from priority scores
          probabilities = [priority_scores[cond[0]] for cond in conditions]
          total_prob = sum(probabilities)
          probabilities = [p / total_prob for p in probabilities]  # Normalize probabilities

          # Randomly select one condition to keep, based on the calculated probabilities
          selected_condition = torch.multinomial(torch.tensor(probabilities), 1).item()

          # Set all other conditions' masks to False
          for idx, cond in enumerate(conditions):
              if idx != selected_condition:
                  cond_type, b, c = cond
                  all_cond[cond_type]['mask'][b, c] = False

  return all_cond

class ConditionGenerator:
  def __init__(self, cond_cfg, split):
    self.split = split
    self.cond_cfg = cond_cfg
    self.cond_funcs = {k: condition_funcs[k] for k in cond_cfg.TYPES}
    self._load_cond_data()

  def _load_cond_data(self):
    print(self.cond_cfg.TYPES)
    print(self.cond_cfg.LLM_TEXT_FOLDER[self.split.upper()])

    if 'motion_tag_text' in self.cond_cfg.TYPES or 'motion_tag_OneText' in self.cond_cfg.TYPES:
      template_file = self.cond_cfg.MOTION_TAG_TEMPLATE
      with open(template_file, 'r') as f:
        self.motion_tag_template = json.load(f)

    if ('llm_text' in self.cond_cfg.TYPES or 'llm_text_OneText' in self.cond_cfg.TYPES) and ('waymo' not in self.cond_cfg.LLM_TEXT_FOLDER[self.split.upper()]):
      llm_text_folder = self.cond_cfg.LLM_TEXT_FOLDER[self.split.upper()]
      self.llm_texts = load_all_llm_texts(llm_text_folder)
    else:
      self.llm_texts = None

  def _sample_condition(self, cond_dict: dict, split:str):
    sample_mode = self.cond_cfg.SAMPLE_MODE[split.upper()]
    
    sample_rate = self.cond_cfg.SAMPLE_RATE
    random_shuffle = self.cond_cfg.RANDOM_SAMPLE[split.upper()]

    for cond_type in cond_dict:
      cond_dict[cond_type] = sample_cond_data(cond_dict[cond_type], sample_method=sample_mode, sample_rate=sample_rate, random_shuffle=random_shuffle, max_cond_per_batch=self.cond_cfg.MAX_COND_PER_BATCH, max_cond_per_scene=self.cond_cfg.MAX_COND_PER_SCENE)

    return cond_dict

  def _obtain_prompt_mask(self, batch, cond_dict):
    '''
    From each of the [B, C, A] prompt idx in the cond_dict, obtain the [B, N] prompt mask indicating whether the prompt agent has been conditioned on.
    Note: A could be different for each condition type; When A > 1, this condition is applied to multiple agents.
    '''
    tgt_agent_idxs = batch.tgt_agent_idxs
    B = len(tgt_agent_idxs)
    N = max([len(tgt_idx) for tgt_idx in tgt_agent_idxs])

    for cond_type in cond_dict:
      goal_mask = cond_dict[cond_type]['mask'] # [B, C]
      prompt_idx = cond_dict[cond_type]['prompt_idx'] # [B, C, A]

      A = prompt_idx.shape[2]
      prompt_mask = torch.zeros((B, N), device=prompt_idx.device).to(torch.bool) # [B, N]

      # iterate over condition index (e.g., agent_1, agent_2)
      for aidx in range(A):
        # obtain the i-th prompt index
        nidx = prompt_idx[:, :, aidx] # [B, C]

        # Create bidx
        bidx = torch.arange(B, device=prompt_idx.device).unsqueeze(1).expand_as(nidx) # [B, C]

        # Advanced indexing to map the mask values
        vmask = (nidx < N) & (nidx >= 0) # [B, C]

        if vmask.sum() == 0:
          continue

        prompt_mask[bidx[vmask], nidx[vmask]] |= goal_mask[vmask]
      
      cond_dict[cond_type]['prompt_mask'] = prompt_mask
    
    return cond_dict

  def _obtain_prompt_str(self, batch, cond_dict):
    for cond_type in cond_dict:
      cond_dict[cond_type]['caption_str'] = caption_funcs[cond_type](batch, cond_dict[cond_type])

    return cond_dict

  def get_condition_from_option(self, cond_type, agent_cond_dict, device):
    return cond_from_option_funcs[cond_type](agent_cond_dict, self.cond_cfg, device)

  def get_batch_condition(self, batch, split):
    '''
    Input: 
      batch: SceneBatch
      split: str - 'train' / 'val' / 'test' / 'rollout'

    Output: BatchCondition
    '''

    all_cond = {}
    for cond_type, cond_func in self.cond_funcs.items():
      if 'motion_tag_text' in cond_type or 'motion_tag_OneText' in cond_type:
        all_cond[cond_type] = cond_func(batch, self.cond_cfg, split, self.motion_tag_template)
      elif 'llm_text' in cond_type:
        all_cond[cond_type] = cond_func(batch, self.cond_cfg, self.llm_texts)
      else:
        all_cond[cond_type] = cond_func(batch, self.cond_cfg)
    
    if self.cond_cfg.USE_PRIORITY_MASK:
      if self.cond_cfg.SAMPLE_BEFORE_PRIORITY:
        all_cond = self._sample_condition(all_cond, split)
      
      if self.cond_cfg.USE_SOFT_PRIORITY:
        cond_priority_scores = dict(self.cond_cfg.PRIORITY_SCORES)
        assert all(cond_type in cond_priority_scores for cond_type in self.cond_funcs), 'not all conditions are in priority list'
        all_cond = _mask_soft_priority_condition(all_cond, cond_priority_scores)
      else:
        cond_priority_order = self.cond_cfg.PRIORITY_ORDER
        assert all(cond_type in cond_priority_order for cond_type in self.cond_funcs), 'not all conditions are in priority list'
        all_cond = _mask_priority_condition(all_cond, cond_priority_order)

    if not (self.cond_cfg.USE_PRIORITY_MASK and self.cond_cfg.SAMPLE_BEFORE_PRIORITY):
      all_cond = self._sample_condition(all_cond, split)
    
    all_cond = self._obtain_prompt_mask(batch, all_cond)

    for cond_type in all_cond:
      if 'OneText' in cond_type:
        all_cond[cond_type] = concat_text_to_OneText(batch, all_cond[cond_type], self.cond_cfg, cond_type)

    all_cond = self._obtain_prompt_str(batch, all_cond)

    return BatchCondition(all_cond)