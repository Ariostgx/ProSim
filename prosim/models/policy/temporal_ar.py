import numpy as np

import torch
import torch.nn as nn
from .act_decoder import AttnRelPE

class TemporalARDecoder(AttnRelPE):
  def _plain_batch_to_temporal(self, batch, pair_names):
    '''
    convert the plain batch to temporal batch according to the pair_names
    
    Input: Batch of shape [B, D], pair_names of shape [B]
    Output: Batch of shape [N, T, D], batch_idx of shape [2, B]
    
    shape change: [B, D] -> [N, T, D]
    B: original batch size (all avaliable agents in avaliable time steps)
    N: the number of pairs
    T: max number of time steps
    B <= N * T (some agents may not have T time steps)
    '''

    agent_names = ['-'.join(name.split('-')[:-1]) for name in pair_names]
    time_steps = [int(name.split('-')[-1]) for name in pair_names]

    unique_agent_names = sorted(list(set(agent_names)))
    unique_time_steps = sorted(list(set(time_steps)))

    N = len(unique_agent_names)
    T = len(unique_time_steps)

    D = batch.shape[-1]
    batch_T = torch.zeros(N, T, D, dtype=batch.dtype, device=batch.device)

    agent_idx = [unique_agent_names.index(name) for name in agent_names]
    time_idx = [unique_time_steps.index(time) for time in time_steps]

    batch_T[agent_idx, time_idx] = batch

    return batch_T, [agent_idx, time_idx]

  def forward(self, policy_emd, batch_obs, batch_map, batch_pos, pair_names, latent_state):
    from prosim.rollout.distributed_utils import get_gpu_memory_usage

    context_emd = self._extract_context(policy_emd)
    policy_batch_idx = policy_emd['batch_idx']
    fuse_feature = self.attn_fuse(context_emd, policy_batch_idx, batch_obs, batch_map, batch_pos)
    fuse_feature_T, T_idx = self._plain_batch_to_temporal(fuse_feature, pair_names)

    pred_feature, latent_state = self._temporal_pred(context_emd, fuse_feature_T, latent_state, T_idx)

    result = self._compute_traj(pred_feature, policy_emd)

    if 'goal' in policy_emd:
      result['goal'] = policy_emd['goal']

    if 'goal_prob' in policy_emd:
      result['goal_prob'] = policy_emd['goal_prob']
      result['goal_point'] = policy_emd['goal_point']
      result['select_idx'] = policy_emd['select_idx']

    result['latent_state'] = latent_state

    return result


class PolicyNoRNN(TemporalARDecoder):
  def _temporal_pred(self, context_emd, fuse_feature_T, latent_state, T_idx):
    pred_feature = fuse_feature_T[T_idx[0], T_idx[1]]
    return pred_feature, latent_state

  def format_latent_state(self, lante_state_dict, all_batch_pair_names):
    
    return None

  def forward(self, policy_emd, batch_obs, batch_map, batch_pos, pair_names, latent_state):
    context_emd = self._extract_context(policy_emd)
    policy_batch_idx = policy_emd['batch_idx']
    fuse_feature = self.attn_fuse(context_emd, policy_batch_idx, batch_obs, batch_map, batch_pos)
    fuse_feature_T, T_idx = self._plain_batch_to_temporal(fuse_feature, pair_names)
    pred_feature, latent_state = self._temporal_pred(None, fuse_feature_T, latent_state, T_idx)

    result = self._compute_traj(pred_feature, policy_emd)

    if 'goal' in policy_emd:
      result['goal'] = policy_emd['goal']
    
    if 'goal_prob' in policy_emd:
      result['goal_prob'] = policy_emd['goal_prob']
      result['goal_point'] = policy_emd['goal_point']

    result['latent_state'] = latent_state
    return result
