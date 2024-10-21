import torch
from trajdata.utils.state_utils import StateTensor

from pytorch_lightning.callbacks import Callback
from prosim.dataset.data_utils import transform_to_frame_offset_rot

def extract_history_obs_from_batch(batch):
  NAN_PADDING = -1e3

  ego_hist = batch.agent_hist.as_format('x,y').float().unsqueeze(1)
  neigh_hist = batch.neigh_hist.as_format('x,y').float()
  
  B = ego_hist.shape[0]
  N = neigh_hist.shape[1]
  device = ego_hist.device

  ego_hist = ego_hist.reshape(B, 1, -1)
  
  if N > 0:
    neigh_hist = neigh_hist.reshape(B, N, -1)
    neigh_mask = torch.zeros([B, N]).to(device).to(torch.bool)
    for b in range(B):
      neigh_mask[b, :batch.num_neigh[b]] = True
      neigh_hist[b, batch.num_neigh[b]:] = NAN_PADDING
    
    neigh_hist[neigh_hist.isnan()] = NAN_PADDING
    
    hist_input = torch.cat([ego_hist, neigh_hist], dim=1)
    hist_mask = torch.cat([torch.ones([B, 1]).to(device).to(torch.bool), neigh_mask], dim=1)
  else:
    hist_input = ego_hist
    hist_mask = torch.ones([B, 1]).to(device).to(torch.bool)

  return hist_input, hist_mask

def extract_agent_obs_from_center_obs(query_names, center_obs):
  agent_obs = {}

  center_name = center_obs.agent_name
  agent_names = center_obs.neigh_names[0] + center_name
  
  control_idx = [agent_names.index(name) for name in query_names]

  obs_format = center_obs.neigh_fut._format
  
  all_hist = torch.concat([center_obs.neigh_hist, center_obs.agent_hist[:, None]], dim=1)
  agent_obs['hist'] = StateTensor.from_array(all_hist[:, control_idx], obs_format)

  T = center_obs.neigh_fut.shape[2]
  all_fut = torch.concat([center_obs.neigh_fut[:, :, :T], center_obs.agent_fut[:, None, :T]], dim=1)
  agent_obs['fut'] = StateTensor.from_array(all_fut[:, control_idx], obs_format)
  
  all_fut_len = torch.concat([center_obs.neigh_fut_len, center_obs.agent_fut_len[:, None]], dim=1)
  agent_obs['fut_len'] = all_fut_len[:, control_idx]

  all_agent_type = torch.concat([center_obs.neigh_types, center_obs.agent_type[:, None]], dim=1)
  agent_obs['type'] = all_agent_type[:, control_idx]

  return agent_obs

def get_agent_pos_dict(agent_hist):
  result = {}

  device = agent_hist.device

  agent_curr = agent_hist[:, :, -1]

  result['position'] = agent_curr.as_format('x,y').float().clone()
  result['heading'] = agent_curr.as_format('h').float().clone()

  return result