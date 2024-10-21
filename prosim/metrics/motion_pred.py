import torch
import torch.nn as nn
import numpy as np
from torchmetrics import Accuracy, MeanMetric, Metric
from prosim.dataset.data_utils import rotate

from prosim.core.registry import registry
from prosim.loss.loss_func import pair_names_to_indices, rollout_temp_traj_preds

class MotionPred(Metric):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.traj_metrics = {}
    self.traj_metrics['ade'] = MeanMetric()
    self.traj_metrics['fde'] = MeanMetric()
    self.traj_metrics['min_ade'] = MeanMetric()
    self.traj_metrics['min_fde'] = MeanMetric()
  
  def compute(self):
    results = {}
    for attr in self.traj_metrics:
      self.traj_metrics[attr].to(self.device)
      results[attr] = self.traj_metrics[attr].compute()
    return results

  def reset(self):
    for attr in self.traj_metrics:
      self.traj_metrics[attr].reset()
  
  def _update_traj_error(self, batch_pred, batch_prob, batch_tgt):
    import time

    B, N, K, T, _ = batch_pred.shape
    K = batch_prob.shape[-1]
    batch_mask = ~(batch_tgt.isnan())

    # Reshape for batch operation
    batch_pred = batch_pred[..., :2]
    batch_tgt = batch_tgt[..., :2]
    batch_mask = batch_mask[..., :2]

    # Get top predictions
    k_index = torch.argmax(batch_prob, dim=-1).reshape(-1)
    valid_mask = ~batch_tgt.isnan().all(dim=-1).reshape(-1, T)

    # Calculate min distance error
    tgt_K = batch_tgt.unsqueeze(2).repeat(1, 1, K, 1, 1)
    valid_mask_T = valid_mask.unsqueeze(1).repeat(1, K, 1)
    dist = (tgt_K - batch_pred).norm(dim=-1).reshape(-1, K, T)
    dist_masked = dist.masked_fill(~valid_mask_T, 0.0)
    ade_K = dist_masked.sum(dim=-1) / valid_mask_T.sum(dim=-1)

    start = time.time()
    ade = ade_K[torch.arange(ade_K.size(0)), k_index]
    min_ade= ade_K.min(dim=-1)[0]
    best_k_index = torch.argmin(ade_K, dim=-1)

    metric_device = self.traj_metrics['ade'].device
    
    self.traj_metrics['ade'].update(ade.to(metric_device))
    self.traj_metrics['min_ade'].update(min_ade.to(metric_device))

    indices = torch.arange(valid_mask.size(1), device=valid_mask.device).unsqueeze(0).expand_as(valid_mask)
    masked_indices = torch.where(valid_mask, indices, torch.tensor(-1, device=valid_mask.device))
    last_valid_idx = masked_indices.max(dim=1).values
    fde_K = dist_masked[torch.arange(dist_masked.size(0)), :, last_valid_idx]

    fde = fde_K[torch.arange(fde_K.size(0)), k_index]
    min_fde = fde_K.min(dim=-1)[0]

    self.traj_metrics['fde'].update(fde.to(metric_device))
    self.traj_metrics['min_fde'].update(min_fde.to(metric_device))

    return k_index, best_k_index

@registry.register_metric(name='ego_traj_pred')
class EgoMotionPred(MotionPred):
  def __init__(self, cfg):
    super().__init__(cfg)
  
  def update(self, batch, output):
    motion_pred = output['motion_pred'].detach()
    motion_prob = output['motion_prob'].detach()
    tgt = batch.agent_fut.as_format('x,y').float().unsqueeze(1)
    self._update_traj_error(motion_pred, motion_prob, tgt)

@registry.register_metric(name='all_traj_pred')
class AllMotionPred(MotionPred):
  def __init__(self, cfg):
    super().__init__(cfg)
  
  def update(self, batch, output):
    motion_pred = output['motion_pred'].detach()
    motion_prob = output['motion_prob'].detach()
    
    tgt = batch.agent_fut.as_format('x,y').float().unsqueeze(1)
    
    if torch.any(batch.num_neigh > 0):
      abs_traj = batch.neigh_fut.as_format('x,y').float()
      heading = batch.neigh_hist[:, :, [-1]].as_format('h').squeeze(-1)
      position = batch.neigh_hist[:, :, [-1]].as_format('x,y')
      rel_traj = abs_traj - position
      rel_traj = rotate(rel_traj[..., 0], rel_traj[..., 1], -heading)
      tgt = torch.cat([tgt, rel_traj], dim=1)

    self._update_traj_error(motion_pred, motion_prob, tgt)

@registry.register_metric(name='pair_traj_pred')
class PairMotionPred(MotionPred):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.traj_metrics['rollout_ade'] = MeanMetric()
    # self.traj_metrics['rollout_min_ade'] = MeanMetric()
    cond_types = cfg.PROMPT.CONDITION.TYPES
    
    for cond_type in cond_types:
      self.traj_metrics[f'rollout_ade_condition_{cond_type}'] = MeanMetric()
    
    if len(cond_types) > 0:
      self.traj_metrics['rollout_ade_condition_none'] = MeanMetric()

  
  def _compute_traj_ade(self, tgt_rollout, pred_rollout, valid_mask):
    # Calculate min distance error over rollout
    # tgt_rollout: [B, N, T, 3]
    # pred_rollout: [B, N, T, 3]
    # valid_mask: [B, N, T, 3]

    dist = (tgt_rollout[..., :2] - pred_rollout[..., :2]).norm(dim=-1) # [B, N, T]
    
    step_valid = valid_mask[..., :2].all(dim=-1) # [B, N, T]
    dist_masked = dist.masked_fill(~step_valid, 0.0) # [B, N, T]

    # average over time
    step_mean_ade = dist_masked.sum(dim=-1) / torch.clamp_min(step_valid.sum(dim=-1), min=1.0) # [B, N]

    # average over agent
    agent_valid = step_valid.any(dim=-1) # [B, N]
    agent_mean_ade = step_mean_ade[agent_valid].mean()

    return agent_mean_ade

  def _update_rollout_ade(self, batch, output, cfg, k_index, best_k_index):
    # not using best_k_index for now
    tgt_rollout_traj, pred_rollout_traj, _, valid_mask, _, _ = rollout_temp_traj_preds(batch, output, cfg, k_index, None)

    metric_device = self.traj_metrics['ade'].device
    tgt_rollout_traj = tgt_rollout_traj.detach().to(metric_device)
    pred_rollout_traj = pred_rollout_traj.detach().to(metric_device)
    valid_mask = valid_mask.detach().to(metric_device)

    ade = self._compute_traj_ade(tgt_rollout_traj, pred_rollout_traj, valid_mask)
    self.traj_metrics['rollout_ade'].update(ade.to(metric_device))

    # compute condition-wise ADE
    batch_cond = batch.extras['condition']
    if len(batch_cond) == 0:
      return
    agent_cond_mask_union = torch.zeros_like(valid_mask[:, :, 0, 0], device=metric_device) # [B, N]
    cond_masks = {}
    for cond_type in batch_cond.keys():
      cond_masks[cond_type] = batch_cond[cond_type]['prompt_mask'].to(metric_device) # [B, N]
      agent_cond_mask_union |= cond_masks[cond_type]

    cond_masks['none'] = ~agent_cond_mask_union

    for cond_type, cond_mask in cond_masks.items():
      cond_valid_mask = valid_mask & cond_mask.unsqueeze(-1).unsqueeze(-1)
      if not cond_valid_mask.any():
        continue
      ade_condition = self._compute_traj_ade(tgt_rollout_traj, pred_rollout_traj, cond_valid_mask)
      # print(ade_condition.device)
      self.traj_metrics[f'rollout_ade_condition_{cond_type}'].update(ade_condition.to(metric_device))

    # best_pred_rollout_traj = best_pred_rollout_traj.detach()
    # min_ade = self._compute_traj_ade(tgt_rollout_traj, best_pred_rollout_traj, valid_mask)
    # self.traj_metrics['rollout_min_ade'].update(min_ade.to(metric_device))
    # print('tgt_rollout_traj', tgt_rollout_traj)
    # print('pred_rollout_traj', pred_rollout_traj)


  def update(self, batch, output):
    if 'motion_pred' in output:
      motion_pred = output['motion_pred'].detach().unsqueeze(0)
      motion_prob = output['motion_prob'].detach().unsqueeze(0)

      pair_names = output['pair_names']
      bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
      tgt = batch.extras['io_pairs_batch']['tgt'][bidxs, tidxs, nidxs][None, :].detach()

      device = self.device
      motion_prob = motion_prob.to(device)
      tgt = tgt.to(device)

      k_index, best_k_index = self._update_traj_error(motion_pred, motion_prob, tgt)

      self._update_rollout_ade(batch, output, self.cfg, k_index, best_k_index)
