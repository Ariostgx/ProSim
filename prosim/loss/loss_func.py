import os
import time
import torch
from typing import Tuple

from prosim.dataset.data_utils import rotate
from prosim.models.utils.geometry import batch_rotate_2D
from prosim.models.utils.geometry import wrap_angle, batch_rotate_2D

def get_min_dist_idx(tgt_gt, pred, crit_func, mask):
  t_mask = mask.all(dim=-1)
  indices = torch.arange(t_mask.size(1), device=mask.device)[None, :].expand_as(t_mask)
  masked_indices = torch.where(t_mask, indices, torch.tensor(-1, device=mask.device))
  last_valid_idx = masked_indices.max(dim=1).values
  tgt_end = tgt_gt[torch.arange(tgt_gt.size(0)), :, last_valid_idx, :2]
  src_end = pred[torch.arange(pred.size(0)), :, last_valid_idx, :2]
  dists = crit_func(tgt_end, src_end).mean(-1)
  min_index = torch.argmin(dists, dim=-1)

  return min_index

def compute_pos_loss(tgt_gt_traj, pred_traj, mask, min_index, crit_func):
  tgt_gt_traj = tgt_gt_traj[:, 0] # [B, T, 3]
  
  bidx = torch.arange(tgt_gt_traj.size(0), device=tgt_gt_traj.device)
  pred_traj = pred_traj[bidx, min_index] # [B, T, 3]

  pos_loss = crit_func(tgt_gt_traj[..., :2], pred_traj[..., :2])
  pos_mask = mask[..., :2]
  pos_loss[~pos_mask] = 0

  pos_loss = pos_loss.sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6) * 2 # [B]
  pos_loss = pos_loss.mean()

  return pos_loss

def compute_gmm_dist(tgt_gt_traj, pred_traj, pred_gmm_params, log_std_range, rho_limit):
  res_trajs = tgt_gt_traj[..., :2] - pred_traj[..., :2] # [B, T, 2]
  dx = res_trajs[..., 0] # [B, T]
  dy = res_trajs[..., 1] # [B, T]

  log_std1 = torch.clip(pred_gmm_params[..., 0], min=log_std_range[0], max=log_std_range[1])
  log_std2 = torch.clip(pred_gmm_params[..., 1], min=log_std_range[0], max=log_std_range[1])
  std1 = torch.exp(log_std1)  # (0.2m to 150m)
  std2 = torch.exp(log_std2)  # (0.2m to 150m)
  rho = torch.clip(pred_gmm_params[..., 2], min=-rho_limit, max=rho_limit)

  # -log(a^-1 * e^b) = log(a) - b
  reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # [B, T]
  reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # [B, T]

  reg_gmm_dist = reg_gmm_log_coefficient + reg_gmm_exp  # [B, T]

  return reg_gmm_dist


def compute_pos_loss_gmm(tgt_gt_traj, pred_traj, pred_gmm_params, mask, min_index, log_std_range=(-1.609, 5.0), rho_limit=0.5):
  # mainly copied from https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/loss_utils.py
  # Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
  # Written by Shaoshuai Shi 
  tgt_gt_traj = tgt_gt_traj[:, 0] # [B, T, 3]

  bidx = torch.arange(tgt_gt_traj.size(0), device=tgt_gt_traj.device)
  pred_traj = pred_traj[bidx, min_index] # [B, T, 3]
  pred_gmm_params = pred_gmm_params[bidx, min_index] # [B, T, 3]

  reg_gmm_loss = compute_gmm_dist(tgt_gt_traj, pred_traj, pred_gmm_params, log_std_range, rho_limit) # [B, T]

  pos_mask = mask[..., :2].all(dim=-1) # [B, T]
  reg_gmm_loss[~pos_mask] = 0

  reg_gmm_loss = reg_gmm_loss.sum(-1) / (pos_mask.sum(-1) + 1e-6) # [B]
  reg_gmm_loss = reg_gmm_loss.mean()

  return reg_gmm_loss


def compute_head_loss(tgt_gt_traj, pred_traj, mask, min_index):
  # use L1 loss for heading
  l1_loss = torch.nn.L1Loss(reduction='none')

  tgt_gt_traj = tgt_gt_traj[:, 0] # [B, T, 3]
  bidx = torch.arange(tgt_gt_traj.size(0), device=tgt_gt_traj.device)
  pred_traj = pred_traj[bidx, min_index] # [B, T, 3]
  
  tgt_heading = torch.stack([torch.sin(tgt_gt_traj[..., 2]), torch.cos(tgt_gt_traj[..., 2])], dim=-1)
  pred_heading = torch.stack([torch.sin(pred_traj[..., 2]), torch.cos(pred_traj[..., 2])], dim=-1)

  head_loss = l1_loss(tgt_heading, pred_heading) # [B, T, 2]

  head_mask = mask[..., 2, None].repeat(1, 1, 2) # [B, T, 2]

  head_loss[~head_mask] = 0
  # print('head_loss', head_loss)
  head_loss = head_loss.sum(-1).sum(-1) / (head_mask.sum(-1).sum(-1) + 1e-6) * 2 # [B]
  head_loss = head_loss.mean()

  return head_loss

def compute_vel_loss(tgt_gt_vel, pred_vel, mask, min_index):
  # use L1 loss for velocity
  l1_loss = torch.nn.L1Loss(reduction='none')

  tgt_gt_vel = tgt_gt_vel[:, 0] # [B, T, 2]
  bidx = torch.arange(tgt_gt_vel.size(0), device=tgt_gt_vel.device)
  pred_vel = pred_vel[bidx, min_index] # [B, T, 2]
  
  vel_loss = l1_loss(tgt_gt_vel, pred_vel) # [B, T, 2]

  vel_mask = mask[..., 3:5] # [B, T, 2]

  vel_loss[~vel_mask] = 0
  vel_loss = vel_loss.sum(-1).sum(-1) / (vel_mask.sum(-1).sum(-1) + 1e-6) * 2 # [B]
  vel_loss = vel_loss.mean()

  return vel_loss

def compute_step_loss(tgt, pred, prob, mask, config):
  to_pred_gmm = config.MODEL.POLICY.ACT_DECODER.TRAJ.PRED_GMM
  to_pred_vel = config.MODEL.POLICY.ACT_DECODER.TRAJ.PRED_VEL
  
  CLS = torch.nn.CrossEntropyLoss(reduction='none')

  if config.LOSS.TRAJ_CRITERION.TYPE == 'mse':
    crit_func = torch.nn.MSELoss(reduction='none')
  elif config.LOSS.TRAJ_CRITERION.TYPE == 'huber':
    crit_func = torch.nn.HuberLoss(reduction='none', delta=config.LOSS.TRAJ_CRITERION.HUBER_DELTA)

  K = pred.shape[1]
  tgt_gt = tgt.unsqueeze(1).repeat(1, K, 1, 1)
  k_mask = mask.unsqueeze(1).repeat(1, K, 1, 1)
  tgt_gt[~k_mask] = 0

  tgt_gt_traj = tgt_gt[..., :3] # [B, K, T, 3]
  tgt_gt_vel = tgt_gt[..., 3:] # [B, K, T, 2] if available, else [B, K, T, 0]

  pred_traj = pred[..., :3] # [B, K, T, 3]
  if to_pred_gmm:
    pred_gmm_params = pred[..., 3:6] # [B, K, T, 3]
    pred_vel = pred[..., 6:] # [B, K, T, 2] if available, else [B, K, T, 0]
  else:
    pred_vel = pred[..., 3:] # [B, K, T, 2] if available, else [B, K, T, 0]
  
  min_index = get_min_dist_idx(tgt_gt_traj, pred_traj, crit_func, mask)

  # compute the position loss
  if to_pred_gmm:
    pos_loss = compute_pos_loss_gmm(tgt_gt_traj, pred_traj, pred_gmm_params, mask, min_index)
  else:
    pos_loss = compute_pos_loss(tgt_gt_traj, pred_traj, mask, min_index, crit_func)

  # compute the heading loss
  head_loss = compute_head_loss(tgt_gt_traj, pred_traj, mask, min_index)

  cls_mask = mask[..., 0].any(dim=-1) # [B]
  cls_loss = CLS(prob, min_index)[cls_mask].mean()

  pos_loss = pos_loss * config.LOSS.STEP_TRAJ.POS_WEIGHT
  cls_loss = cls_loss * config.LOSS.STEP_TRAJ.CLS_WEIGHT
  head_loss = head_loss * config.LOSS.STEP_TRAJ.HEAD_WEIGHT
  full_loss = pos_loss + cls_loss + head_loss

  result = {'pos_loss': pos_loss, 'cls_loss': cls_loss, 'head_loss': head_loss, 'full_loss': full_loss}

  if to_pred_vel:
    vel_loss = compute_vel_loss(tgt_gt_vel, pred_vel, mask, min_index)
    vel_loss = vel_loss * config.LOSS.STEP_TRAJ.VEL_WEIGHT
    result['vel_loss'] = vel_loss
    result['full_loss'] += vel_loss
  
  return result, min_index
  
def ego_mse_k_way(batch, model_output, config):
  tgt = batch.agent_fut.as_format('x,y').float().unsqueeze(1)
  pred = model_output['motion_pred']
  prob = model_output['motion_prob']
  mask = ~(tgt.isnan())

  return compute_step_loss(tgt, pred, prob, mask, config)

def empty(batch, model_output, config):
  return {'full_loss': torch.tensor(0.0)}


def motion_mse_k_way(batch, model_output, config):
  ego_tgt = batch.agent_fut.as_format('x,y').float().unsqueeze(1)
  
  # transform neigh_tgt to relative trajecotry
  abs_traj = batch.neigh_fut.as_format('x,y').float()
  heading = batch.neigh_hist[:, :, [-1]].as_format('h').squeeze(-1)
  position = batch.neigh_hist[:, :, [-1]].as_format('x,y')
  rel_traj = abs_traj - position
  rel_traj = rotate(rel_traj[..., 0], rel_traj[..., 1], -heading)

  tgt = torch.cat([ego_tgt, rel_traj], dim=1)
  
  pred = model_output['motion_pred']
  prob = model_output['motion_prob']
  mask = ~(tgt.isnan())

  try:
    return compute_step_loss(tgt, pred, prob, mask, config)
  except Exception as e:
    print(e)
    save_path = os.path.join(config.SAVE_DIR, config.EXPERIMENT_NAME)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'error_batch.pt')
    batch.to('cpu')
    save_dict = {'batch': batch, 'motion_pred': pred.detach().cpu(), 'motion_prob': prob.detach().cpu(), 'mask': mask.detach().cpu()}
    torch.save(save_dict, save_file)
    print('error computing motion_mse_k_way')
    print('saved error batch to {}'.format(save_file))
    return torch.tensor(0.0).to(pred.device)

def rollout_traj(traj, rollout_steps):
  # traj: [B, N, T, pred_steps, 3/5]

  B, N, T, pred_steps, D = traj.shape

  to_pred_vel = D == 5

  dtheta = traj[..., rollout_steps-1, 2]
  theta = torch.cumsum(dtheta, dim=-1)
  theta = torch.cat([torch.zeros_like(theta[..., :1]), theta[..., :-1]], dim=-1)
  theta = wrap_angle(theta)

  dx = torch.diff(traj[..., :2], dim=-2)
  dx = torch.cat([traj[..., :1, :2], dx], dim=-2)

  dx_rot = batch_rotate_2D(dx, theta[..., None])
  dx_rot = dx_rot[..., :rollout_steps, :].reshape(B, N, -1, 2)

  rollout_traj = torch.cumsum(dx_rot, dim=-2)

  rollout_theta = traj[..., :rollout_steps, 2] + theta[..., None]
  rollout_theta = rollout_theta.reshape(B, N, -1)
  rollout_theta = wrap_angle(rollout_theta)

  result = torch.cat([rollout_traj, rollout_theta[..., None]], dim=-1)

  if to_pred_vel:
    vel = traj[..., :rollout_steps, 3:]
    vel_rot = batch_rotate_2D(vel, theta[..., None])
    vel_rot = vel_rot.reshape(B, N, -1, 2)

    result = torch.cat([result, vel_rot], dim=-1)

  return result

def rollout_temp_traj_preds(batch, output, config, k_index, best_k_index):
  rollout_steps = config.ROLLOUT.POLICY.REPLAN_FREQ
  to_pred_gmm = config.MODEL.POLICY.ACT_DECODER.TRAJ.PRED_GMM
  
  pair_names = output['pair_names']
  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])

  device = output['motion_pred'].device
  dtype = output['motion_pred'].dtype

  D = batch.extras['io_pairs_batch']['tgt'].shape[-1]
  to_pred_vel = D == 5

  tgt_rollout_batch = batch.extras['io_pairs_batch']['tgt'][..., :D].to(device).detach().clone() # [B, T, N, pred_steps, 3/5]

  tgt_rollout_batch = tgt_rollout_batch.permute(0, 2, 1, 3, 4) # [B, N, T, pred_steps, 3/5]
  tgt_mask = batch.extras['io_pairs_batch']['mask'].to(device)
  tgt_mask = tgt_mask.permute(0, 2, 1) # [B, N, T, pred_steps, 2]
  
  valid_mask =  tgt_mask[..., None, None] * (~tgt_rollout_batch.isnan())

  tgt_rollout_batch[~valid_mask] = 0.0

  A = k_index.size(0)

  pred_index = torch.arange(A, device=device)

  pred_rollout_batch = {}
  gmm_params_batch = {}
  k_indices = {'pred': k_index, 'best': best_k_index}
  for key in k_indices:
    k_index = k_indices[key]

    if k_index is None:
      continue

    pred_traj = output['motion_pred'][pred_index, k_index, :, :3]
    if to_pred_vel:
      if to_pred_gmm:
        pred_vel = output['motion_pred'][pred_index, k_index, :, 6:]
      else:
        pred_vel = output['motion_pred'][pred_index, k_index, :, 3:5]
      pred_traj = torch.cat([pred_traj, pred_vel], dim=-1)
    
    pred_rollout_batch[key] = torch.zeros_like(tgt_rollout_batch, device=device, dtype=dtype) # [B, N, T, pred_steps, 3/5]

    pred_rollout_batch[key][bidxs, nidxs, tidxs] = pred_traj

    if to_pred_gmm:
      B, N, T = tgt_rollout_batch.shape[:3]
      gmm_params_batch[key] = torch.zeros(B, N, T, rollout_steps, 3, device=device) # [B, N, T, rollout_steps, 3]
      gmm_params_batch[key][bidxs, nidxs, tidxs] = output['motion_pred'][pred_index, k_index, :rollout_steps, 3:6] # [B, N, T, rollout_steps, 3]
      gmm_params_batch[key] = gmm_params_batch[key].reshape(B, N, T*rollout_steps, 3)
    else:
      gmm_params_batch[key] = None

  tgt_rollout_traj = rollout_traj(tgt_rollout_batch, rollout_steps)
  pred_rollout_traj = rollout_traj(pred_rollout_batch['pred'], rollout_steps)
  # best_pred_rollout_traj = rollout_traj(pred_rollout_batch['best'], rollout_steps)

  B, N, full_steps, _ = tgt_rollout_traj.shape
  valid_mask = valid_mask[..., :rollout_steps, :].reshape(B, N, full_steps, D)

  return tgt_rollout_traj, pred_rollout_traj, None, valid_mask, gmm_params_batch['pred'], None

def compute_rollout_loss(tgt_rollout_traj, model_rollout_traj, valid_mask, config, model_gmm_params_batch=None):
  if config.LOSS.TRAJ_CRITERION.TYPE == 'mse':
    crit_func = torch.nn.MSELoss(reduction='none')
  elif config.LOSS.TRAJ_CRITERION.TYPE == 'huber':
    crit_func = torch.nn.HuberLoss(reduction='none', delta=config.LOSS.TRAJ_CRITERION.HUBER_DELTA)
  
  l1_loss = torch.nn.L1Loss(reduction='none')

  if model_gmm_params_batch is None:
    pos_dist = crit_func(tgt_rollout_traj[..., :2], model_rollout_traj[..., :2]).sum(dim=-1) # [B, N, T]
  else:
    pos_dist = compute_gmm_dist(tgt_rollout_traj, model_rollout_traj, model_gmm_params_batch, (-1.609, 5.0), 0.5) # [B, N, T]
  
  tgt_heading = torch.stack([torch.sin(tgt_rollout_traj[..., 2]), torch.cos(tgt_rollout_traj[..., 2])], dim=-1)
  model_heading = torch.stack([torch.sin(model_rollout_traj[..., 2]), torch.cos(model_rollout_traj[..., 2])], dim=-1)
  head_dist = l1_loss(tgt_heading, model_heading).sum(-1) # [B, N, T]

  dists = {'pos': pos_dist, 'heading': head_dist}

  # predict velocity
  if tgt_rollout_traj.shape[-1] == 5:
    tgt_vel = tgt_rollout_traj[..., 3:]
    model_vel = model_rollout_traj[..., 3:]
    vel_dist = crit_func(tgt_vel, model_vel).sum(dim=-1) # [B, N, T]

    dists['vel'] = vel_dist


  step_valid = valid_mask[..., :2].all(dim=-1) # [B, N, T]
  agent_valid = step_valid.any(dim=-1) # [B, N]

  loss = {}
  loss_per_agent = {}
  for key in dists:
    dist_masked = dists[key].masked_fill(~step_valid, 0.0) # [B, N, T]

    # average over time
    step_mean_dist = dist_masked.sum(dim=-1) / torch.clamp_min(step_valid.sum(dim=-1), min=1.0) # [B, N]

    # average over agent
    agent_mean_dist = step_mean_dist[agent_valid].mean()
    loss[key] = agent_mean_dist

    loss_per_agent[key] = step_mean_dist
  loss_per_agent['agent_valid_mask'] = agent_valid

  return loss, loss_per_agent

def pair_names_to_indices(pair_names, io_pairs_batch):
  agent_names = io_pairs_batch['agent_names']
  T_indices = io_pairs_batch['T_indices']

  bidxs, tidxs, nidxs = [], [], []
  for pair_name in pair_names:
    bidx, agent_name, tidx = pair_name.split('-')
    bidxs.append(int(bidx))
    tidxs.append(T_indices.index(int(tidx)))
    nidxs.append(agent_names[int(bidx)].index(agent_name))
  
  return bidxs, tidxs, nidxs

def compute_condition_type_rloss(batch, loss_result, rloss_per_agent):
  batch_cond = batch.extras['condition']
  if len(batch_cond) == 0:
    return loss_result

  agent_valid_mask = rloss_per_agent['agent_valid_mask'] # [B, N]
  agent_cond_mask_union = torch.zeros_like(agent_valid_mask) # [B, N]
  
  cond_masks = {}
  for cond_type in batch_cond.keys():
    cond_valid_mask = batch_cond[cond_type]['prompt_mask'] # [B, N]
    agent_cond_mask_union |= cond_valid_mask
    cond_masks[cond_type] = cond_valid_mask & agent_valid_mask
  
  cond_masks['none'] = agent_valid_mask & ~agent_cond_mask_union

  for cond_type in cond_masks.keys():
    cond_mask = cond_masks[cond_type]
    if cond_mask.any():
      for loss_type in rloss_per_agent.keys():
        if loss_type == 'agent_valid_mask':
          continue
        loss = rloss_per_agent[loss_type] # [B, N]
        loss_per_cond = loss[cond_mask].mean()
        loss_result[f'conditional_{cond_type}_rollout_{loss_type}_loss'] = loss_per_cond
  
  return loss_result

def paired_mse_k_way(batch, model_output, config):
  pair_names = model_output['pair_names']

  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
  tgt = batch.extras['io_pairs_batch']['tgt'][bidxs, tidxs, nidxs]

  if 'motion_pred' in model_output:
    pred = model_output['motion_pred']
    prob = model_output['motion_prob']
    mask = ~(tgt.isnan())

    if config.LOSS.ROLLOUT_TRAJ.ENABLE:
      results = {'full_loss': torch.tensor(0.0).to(tgt.device)}
    else:
      results, _ = compute_step_loss(tgt, pred, prob, mask, config)

    if config.LOSS.ROLLOUT_TRAJ.ENABLE:
      pred_k_index = prob.argmax(dim=-1)
      # best_k_index = batch_traj_idx

      tgt_rollout_traj, pred_rollout_traj, _, rollout_valid_mask, pred_gmm_params, _ = rollout_temp_traj_preds(batch, model_output, config, pred_k_index, None)
      
      model_rollout_traj = pred_rollout_traj
      model_gmm_params = pred_gmm_params
      
      rloss, rloss_per_agent = compute_rollout_loss(tgt_rollout_traj, model_rollout_traj, rollout_valid_mask, config, model_gmm_params)

      results['rollout_pos_loss'] = rloss['pos']
      results['rollout_head_loss'] = rloss['heading']

      r_loss = rloss['pos'] + rloss['heading'] * config.LOSS.ROLLOUT_TRAJ.HEAD_WEIGHT
      if 'vel' in rloss:
        results['rollout_vel_loss'] = rloss['vel']
        r_loss += rloss['vel'] * config.LOSS.ROLLOUT_TRAJ.VEL_WEIGHT

      results['full_loss'] += r_loss * config.LOSS.ROLLOUT_TRAJ.WEIGHT

      with torch.no_grad():
        results = compute_condition_type_rloss(batch, results, rloss_per_agent)

      if config.LOSS.ROLLOUT_TRAJ.USE_OFFROAD_LOSS:
        T_sample_step = config.LOSS.ROLLOUT_TRAJ.OFFROAD_T_SAMPLE_RATE
        offroad_loss = compute_rollout_offroad_loss_batch(batch, model_rollout_traj, tgt_rollout_traj, rollout_valid_mask, tgt_mode = config.LOSS.ROLLOUT_TRAJ.OFFROAD_TGT_MODE, T_sample_step=T_sample_step, M=3)
        results['rollout_offroad_loss'] = offroad_loss
        results['full_loss'] += offroad_loss * config.LOSS.ROLLOUT_TRAJ.OFFROAD_WEIGHT
      
      if config.LOSS.ROLLOUT_TRAJ.USE_COLLISION_LOSS:
        T_sample_step = config.LOSS.ROLLOUT_TRAJ.COLLISION_T_SAMPLE_RATE
        tgt_mode = config.LOSS.ROLLOUT_TRAJ.COLLISION_TGT_MODE
        K = config.LOSS.ROLLOUT_TRAJ.COLLISION_K
        collision_threshold = config.LOSS.ROLLOUT_TRAJ.COLLISION_THRESHOLD
        vehicle_only = config.LOSS.ROLLOUT_TRAJ.COLLISION_VEHICLE_ONLY

        collision_loss = compute_rollout_collision_loss_batch(batch, model_rollout_traj, tgt_rollout_traj, rollout_valid_mask, tgt_mode, T_sample_step=T_sample_step, K = K, collision_threshold=collision_threshold, vehicle_only=vehicle_only)
        results['rollout_collision_loss'] = collision_loss
        results['full_loss'] += collision_loss * config.LOSS.ROLLOUT_TRAJ.COLLISION_WEIGHT
    
    if config.LOSS.GOAL_DIST_PRED.ENABLE:
      goal_prob_losses = goal_prob_pred(batch, model_output, config)
      results.update(goal_prob_losses)
      results['full_loss'] += goal_prob_losses['goal_dist_all'] * config.LOSS.GOAL_DIST_PRED.WEIGHT
        
  else:
    results = {'full_loss': 0.0}
  
  if config.LOSS.ROLLOUT_TRAJ.USE_GOAL_PRED_LOSS:
    goal_loss_dict = goal_pred(batch, model_output, config)
    results['goal_loss_all'] = 0 
    for key in goal_loss_dict.keys():
      if config.LOSS.ROLLOUT_TRAJ.GOAL_PRED_LOSS_COND_MASK and 'uncond' in key:
        continue
      results[key] = goal_loss_dict[key]
      results['goal_loss_all'] += goal_loss_dict[key]
  
    results['full_loss'] += results['goal_loss_all'] * config.LOSS.ROLLOUT_TRAJ.GOAL_WEIGHT
  
  if config.LOSS.ROLLOUT_TRAJ.USE_PROMPT_LOSS:
    if 'prompt_loss' in model_output:
      prompt_loss = model_output['prompt_loss']
      if prompt_loss is not None:
        for key in prompt_loss.keys():
          results[key] = prompt_loss[key]
          results['full_loss'] += prompt_loss[key] * config.LOSS.ROLLOUT_TRAJ.PROMPT_WEIGHT
    
  return results

def init_vel_pred(batch, model_output, config):

  pair_names = model_output['pair_names']

  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
  init_vel = batch.extras['io_pairs_batch']['init_vel'][bidxs, tidxs, nidxs]
  mask = batch.extras['io_pairs_batch']['mask'][bidxs, tidxs, nidxs]

  pred = model_output['reconst_pred']

  crit_func = torch.nn.MSELoss(reduction='mean')

  vel_loss = crit_func(pred[mask], init_vel[mask])

  return {'vel_loss': vel_loss, 'full_loss': vel_loss}

def extend_pred(batch, model_output, config):

  pair_names = model_output['pair_names']

  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
  extend = batch.extras['io_pairs_batch']['extend'][bidxs, tidxs, nidxs]
  mask = batch.extras['io_pairs_batch']['mask'][bidxs, tidxs, nidxs]

  pred = model_output['reconst_pred']

  crit_func = torch.nn.MSELoss(reduction='mean')

  extend_loss = crit_func(pred[mask], extend[mask])

  return {'extend_loss': extend_loss, 'full_loss': extend_loss}


def goal_pred(batch, model_output, config):

  pair_names = model_output['pair_names']

  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
  goal_input = batch.extras['io_pairs_batch']['goal'][bidxs, tidxs, nidxs]

  prompt_mask = batch.extras['io_pairs_batch']['mask'][bidxs, tidxs, nidxs]
  
  cond_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
  for key in ['goal_OneText', 'motion_tag_OneText', 'llm_text_OneText']:
    if key in batch.extras['condition'].keys():
      cond_mask |= batch.extras['condition'][key]['prompt_mask'][bidxs, nidxs]
  
  uncond_mask = prompt_mask & ~cond_mask
  mask_dict = {'cond': cond_mask, 'uncond': uncond_mask}
  goal_loss_dict = {}

  pred = model_output['reconst_pred']
  crit_func = torch.nn.MSELoss(reduction='mean')
  device = pred.device
  t_mask = (torch.tensor(tidxs) == 0).to(device) # only consider the first timestep

  for key in mask_dict.keys():
    mask = mask_dict[key] & t_mask
    if mask.any():
      goal_loss_dict[key+'_goal'] = crit_func(pred[mask], goal_input[mask])
    else:
      goal_loss_dict[key+'_goal'] = torch.tensor(0.0).to(device)
  
  return goal_loss_dict

def goal_prob_pred(batch, model_output, config):
  # compute probability goal prediction loss

  pair_names = model_output['pair_names']

  bidxs, tidxs, nidxs = pair_names_to_indices(pair_names, batch.extras['io_pairs_batch'])
  
  goal_gt = batch.extras['io_pairs_batch']['goal'][bidxs, tidxs, nidxs]
  prompt_mask = batch.extras['io_pairs_batch']['mask'][bidxs, tidxs, nidxs]
  device = goal_gt.device

  t_mask = (torch.tensor(tidxs) == 0).to(device) # only consider the first timestep
  mask = prompt_mask & t_mask

  # filter out the agents that that less timesteps for mroe accurate goal prediction
  oidxs = []
  
  for bidx, nidx in zip(bidxs, nidxs):
    oidxs.append(batch.tgt_agent_idxs[bidx][nidx])
  
  agent_fut_len = batch.agent_fut_len[bidxs, oidxs]
  fut_len_mask = agent_fut_len >= config.LOSS.GOAL_DIST_PRED.MIN_FUT_STEP
  mask = mask & fut_len_mask

  
  goal_pred_point = model_output['goal_point'][mask] # [Q, K, 2]
  goal_pred_prob = model_output['goal_prob'][mask] # [Q, K]

  prob_func = torch.nn.CrossEntropyLoss(reduction='mean')

  # select the closest goal, encourage the model to predict the closest goal
  select_idx = torch.argmin(torch.norm(goal_pred_point - goal_gt[mask][:, None, :2], dim=-1), dim=-1) # [Q]
  prob_loss = prob_func(goal_pred_prob, select_idx)

  # mse loss for the nearest goal point
  point_func = torch.nn.HuberLoss(reduction='mean', delta=config.LOSS.TRAJ_CRITERION.HUBER_DELTA)
  N = goal_pred_point.size(0)
  point_loss = point_func(goal_pred_point[torch.arange(N, device=device), select_idx], goal_gt[mask][:, :2])
  
  # compute varaince of the goal prediction for each agent
  goal_logvar = torch.log(torch.var(goal_pred_point, dim=0) + 1e-6).mean()
  
  with torch.no_grad():
    goal_pred_dist = torch.softmax(goal_pred_prob, dim=-1)
    goal_prob_entropy = -torch.sum(goal_pred_dist * torch.log(goal_pred_dist + 1e-6), dim=-1).mean()
  
  result = {'goal_dist_prob_loss': prob_loss, 'goal_dist_point_loss': point_loss, 'goal_dist_neg_logvar': -goal_logvar, 'goal_dist_entropy': goal_prob_entropy}

  full_loss = point_loss + prob_loss * config.LOSS.GOAL_DIST_PRED.CLS_WEIGHT - goal_logvar * config.LOSS.GOAL_DIST_PRED.VAR_WEIGHT

  result['goal_dist_all'] = full_loss

  return result


def pass_loss(batch, model_output, config):

  loss = model_output['emd']['loss']

  return {'model_loss': loss, 'full_loss': loss}


def _compute_signed_distance_to_polyline(xys, polyline):
  _CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
  """
  Args:
  xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
    query points.
  polyline: A float Tensor of shape (num_segments+1, 2) containing sequences
    of xy coordinates representing start and end points of consecutive
    segments.

  Returns:
  A tensor of shape (num_points), containing the signed distance from queried
    points to the polyline.
  """
  is_cyclic = torch.sum((polyline[0] - polyline[-1]) ** 2) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
  xy_starts = polyline[None, :-1, :2]
  xy_ends = polyline[None, 1:, :2]
  start_to_point = xys[:, None, :2] - xy_starts
  start_to_end = xy_ends - xy_starts

  rel_t = torch.sum(start_to_point * start_to_end, dim=-1) / torch.sum(start_to_end * start_to_end, dim=-1)

  # The cross product in 2D is a scalar and the sign function is used to determine the direction
  n = torch.sign(start_to_point[:, :, 0] * start_to_end[:, :, 1] - start_to_point[:, :, 1] * start_to_end[:, :, 0])

  distance_to_segment = torch.norm(start_to_point - (start_to_end * rel_t[..., None].clamp(0, 1)), dim=-1)

  start_to_end_padded = torch.cat([start_to_end[:, -1:], start_to_end, start_to_end[:, :1]], dim=1)

  cross_prod = start_to_end_padded[:, :-1, 0] * start_to_end_padded[:, 1:, 1] - start_to_end_padded[:, :-1, 1] * start_to_end_padded[:, 1:, 0]
  is_locally_convex = torch.gt(cross_prod, 0.0)

  n_prior = torch.cat([torch.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]], dim=-1)
  n_next = torch.cat([n[:, 1:], torch.where(is_cyclic, n[:, :1], n[:, -1:])], dim=-1)

  sign_if_before = torch.where(is_locally_convex[:, :-1], torch.maximum(n, n_prior), torch.minimum(n, n_prior))
  sign_if_after = torch.where(is_locally_convex[:, 1:], torch.maximum(n, n_next), torch.minimum(n, n_next))

  sign_to_segment = torch.where(rel_t < 0.0, sign_if_before, torch.where(rel_t < 1.0, n, sign_if_after))

  distance_sign = torch.gather(sign_to_segment, 1, torch.argmin(distance_to_segment, dim=-1).unsqueeze(1)).squeeze(1)
  return distance_sign * torch.min(distance_to_segment, dim=-1)[0]


def get_yaw_rotation(yaw):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    rotation_matrix = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    return rotation_matrix

def get_upright_3d_box_corners(boxes):
    # Assuming boxes is a tensor of shape [N, 7]
    center_x, center_y, center_z, length, width, height, heading = boxes.unbind(-1)

    # Get rotation matrices, assuming get_yaw_rotation is defined for PyTorch
    rotation = get_yaw_rotation(heading)  # [N, 3, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # Define corners
    corners = torch.stack([
        l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
        -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
    ], dim=-1).reshape(-1, 8, 3)

    # Apply rotation and translation
    corners = torch.einsum('nij,nkj->nki', rotation, corners) + translation.unsqueeze(-2)

    return corners

OFFROAD_DISTANCE_THRESHOLD = 0.0

def compute_offroad_dist(xys, sizes, headings, polyline_tensors_local):
    '''
    Compute signed offroad dist for each corner of the bounding box for agents in a single batch
    Inputs:
    - xys: (N, T, 2)
    - sizes: (N, T, 2)
    - headings: (N, T, 1)
    - polyline_tensors_local: List([num_points, 2])

    Outputs:
    - offroad_dist: (N, T)
      The most off-road corner per object per timestamp
      Negative: in-road / Positive: off-road

    '''

    N, T = headings.shape[:2]

    box_input = torch.zeros(N, T, 7)
    box_input[:, :, :2] = xys
    box_input[:, :, -1] = headings
    box_input[:, :, 3:5] = sizes

    # (N, T, 4, 2)
    box_input = box_input.reshape(N*T, 7)
    bbox_corners = get_upright_3d_box_corners(box_input)[:, :4, :2]
    bbox_corners = bbox_corners.reshape(N, T, 4, 2)

    # (N*T*4, 2)
    query_points = bbox_corners.reshape(-1, 2)

    polyline_dists = []
    for polyline in polyline_tensors_local:
      if len(polyline) < 2:
        continue
      start = time.time()
      dist = _compute_signed_distance_to_polyline(query_points, polyline.to(query_points.device))
      polyline_dists.append(dist)

    polyline_dists = torch.stack(polyline_dists, dim=-1)
    closest_idx = torch.argmin(torch.abs(polyline_dists), dim=-1)

    # (N*T*4)
    offroad_dist = polyline_dists[torch.arange(N*T*4), closest_idx]
    offroad_dist = offroad_dist.reshape(N, T, 4)

    # reduce to most off-road corner
    # (N, T)
    offroad_dist = offroad_dist.max(dim=-1)[0]

    return offroad_dist

def compute_rollout_offroad_dist_steps(trajs, init_headings, init_xys, sizes, polyline_tensors_local):
    xys_in_center = batch_rotate_2D(trajs[:, :, :2], init_headings) + init_xys
    hs_in_centers = wrap_angle(trajs[:, :, -1] + init_headings)

    T = xys_in_center.shape[1]
    offroad_dist_steps = compute_offroad_dist(xys_in_center, sizes.unsqueeze(1).repeat(1, T, 1), hs_in_centers, polyline_tensors_local)

    return offroad_dist_steps

def compute_rollout_offroad_loss_idx(batch, bidx, rollout_traj, road_edge_polylines):
    init_xys = batch.extras['io_pairs_batch']['position'][bidx, 0].unsqueeze(1) # (N, 1, 2)
    init_headings = batch.extras['io_pairs_batch']['heading'][bidx, 0, :, 0].unsqueeze(1) # (N, 1)
    sizes = batch.extras['io_pairs_batch']['extend'][bidx, 0] # (N, 2)
    agent_types = batch.extras['io_pairs_batch']['agent_type'][bidx, 0] # (N, 1)

    # mask (N): 1 if agent should be on road (vehicle), else 0
    mask = agent_types == 1

    init_point = torch.zeros_like(rollout_traj[bidx][mask][:, :1])
    traj_with_init = torch.cat([init_point, rollout_traj[bidx][mask]], dim=1)

    pred_offroad_dist_steps = compute_rollout_offroad_dist_steps(traj_with_init, init_headings[mask], init_xys[mask], sizes[mask], road_edge_polylines[bidx])
    
    # init_onroad_mask: 1 if agent is initialized on road, else 0
    init_onroad_mask = pred_offroad_dist_steps[:, 0] < 0

    pred_offroad_dist_steps = pred_offroad_dist_steps[init_onroad_mask, 1:]

    # only consider points that are offroad
    pred_offroad_dist_steps = torch.clamp(pred_offroad_dist_steps, min=0.0)

    if pred_offroad_dist_steps.numel() == 0:
      return 0.0

    return pred_offroad_dist_steps.mean()

def _compute_signed_distance_to_polyline_batch(xys, polyline):
  _CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
  """
  Args:
  xys: A float Tensor of shape (Q, num_points, 2) containing xy coordinates of
    query points.
  polyline: A float Tensor of shape (Q, num_segments+1, 2) containing sequences
    of xy coordinates representing start and end points of consecutive
    segments.

  Returns:
  A tensor of shape (Q, num_points), containing the signed distance from queried
    points to the polyline.
  """
  is_cyclic = torch.sum((polyline[..., 0] - polyline[..., -1]) ** 2) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
  xy_starts = polyline[:, None, :-1, :2]
  xy_ends = polyline[:, None, 1:, :2]
  start_to_point = xys[:, :, None, :2] - xy_starts
  start_to_end = xy_ends - xy_starts

  rel_t = torch.sum(start_to_point * start_to_end, dim=-1) / torch.sum(start_to_end * start_to_end, dim=-1)

  # The cross product in 2D is a scalar and the sign function is used to determine the direction
  n = torch.sign(start_to_point[..., 0] * start_to_end[..., 1] - start_to_point[..., 1] * start_to_end[..., 0])

  distance_to_segment = torch.norm(start_to_point - (start_to_end * rel_t[..., None].clamp(0, 1)), dim=-1)

  start_to_end_padded = torch.cat([start_to_end[:, :, -1:], start_to_end, start_to_end[:, :, :1]], dim=2)

  cross_prod = start_to_end_padded[..., :-1, 0] * start_to_end_padded[..., 1:, 1] - start_to_end_padded[..., :-1, 1] * start_to_end_padded[..., 1:, 0]
  is_locally_convex = torch.gt(cross_prod, 0.0)

  n_prior = torch.cat([torch.where(is_cyclic, n[:, :, -1:], n[:, :, :1]), n[:, :, :-1]], dim=-1)
  n_next = torch.cat([n[:, :, 1:], torch.where(is_cyclic, n[:, :, :1], n[:, :, -1:])], dim=-1)

  sign_if_before = torch.where(is_locally_convex[:, :, :-1], torch.maximum(n, n_prior), torch.minimum(n, n_prior))
  sign_if_after = torch.where(is_locally_convex[:, :, 1:], torch.maximum(n, n_next), torch.minimum(n, n_next))

  sign_to_segment = torch.where(rel_t < 0.0, sign_if_before, torch.where(rel_t < 1.0, n, sign_if_after))

  distance_sign = torch.gather(sign_to_segment, 2, torch.argmin(distance_to_segment, dim=-1).unsqueeze(2)).squeeze(1)
  return distance_sign * torch.min(distance_to_segment, dim=-1)[0]

def compute_rollout_offroad_dist_batch(batch, input_rollout_traj, T_sample_step=1, M=3):
  # Compute the offroad distance for every valid agent in the batch
  agent_types = batch.extras['io_pairs_batch']['agent_type'][:, 0] # (B, N, 1)

  rollout_traj = input_rollout_traj[:, :, ::T_sample_step] # (B, N, T, 2)
  B, N, T = rollout_traj.shape[:3]

  # mask (B, N): 1 if agent should be on road (vehicle), else 0
  valid_mask = agent_types == 1 # (B, N)

  A = valid_mask.sum().item() # A is the number of valid agents in the batch
  device = valid_mask.device

  agent_bidx = torch.arange(B, device=device)[:, None].repeat(1, N)[valid_mask]
  agent_nidx = torch.arange(N, device=device)[None].repeat(B, 1)[valid_mask]

  init_pos = torch.zeros_like(rollout_traj[valid_mask][:, :1], device=device) # (A, 1, 2)
  traj_with_init = torch.cat([init_pos, rollout_traj[valid_mask]], dim=1) # (A, T+1, 3)

  init_xy = batch.extras['io_pairs_batch']['position'][:, 0].unsqueeze(2)[valid_mask] # (A, 1, 2)
  init_h = batch.extras['io_pairs_batch']['heading'][:, 0, :, 0].unsqueeze(2)[valid_mask] # (A, 1)
  sizes = batch.extras['io_pairs_batch']['extend'][:, 0][valid_mask] # (A, 2)

  traj_xy_global = batch_rotate_2D(traj_with_init[..., :2], init_h) + init_xy # (A, T+1, 2)
  traj_h_global = wrap_angle(traj_with_init[:, :, -1] + init_h) # (A, T+1)
  traj_sizes = sizes[:, None].repeat(1, T+1, 1) # (A, T+1, 2)
  # Arange the polyline tensors from the batch
  # Obtain b_polyline: [B, P, S, 2] all polyline points in the batch
  # Obtain b_polyline_mask: [B, P, S] valid polyline point mask

  b_polyline = batch.extras['road_edge_polyline'].b_polyline_tensor # [B, P, S, 2]
  b_polyline_mask = ~(b_polyline.isnan().any(dim=-1)) # [B, P, S]
  P, S = b_polyline.shape[1:3]

  # For each of the A agents, obtain the four edges - 
  box_input = torch.zeros(A, T+1, 7, device=device)
  box_input[:, :, :2] = traj_xy_global
  box_input[:, :, -1] = traj_h_global
  box_input[:, :, 3:5] = traj_sizes

  box_input = box_input.reshape(A*(T+1), 7)
  bbox_corners = get_upright_3d_box_corners(box_input)[:, :4, :2]
  bbox_corners = bbox_corners.reshape(A, T+1, 4, 2) # (A, T+1, 4, 2)

  bbox_bidx = agent_bidx[:, None, None].repeat(1, T+1, 4) # (A, T+1, 4)
  bbox_nidx = agent_nidx[:, None, None].repeat(1, T+1, 4) # (A, T+1, 4)

  # flaten the bbox_corners to obtain query_points
  Q = A*(T+1)*4
  query_points = bbox_corners.reshape(Q, 2) # (Q, 2)
  query_bidx = bbox_bidx.reshape(Q) # (Q)
  query_nidx = bbox_nidx.reshape(Q) # (Q)

  # For each query point in (Q, 2), obtain its nearest polyline point from (P, S, 2) polyline tensor
  b_polyline_for_dist = b_polyline[query_bidx] # (Q, P, S, 2)
  q_polyline_mask = b_polyline_mask[query_bidx] # (Q, P, S)

  # Compute distance from each query point to each polyline point
  with torch.no_grad():
    points_to_polyline_dist = (b_polyline_for_dist - query_points[:, None, None])**2 # (Q, P, S, 2)
    points_to_polyline_dist = points_to_polyline_dist.sum(-1) # (Q, P, S)
    points_to_polyline_dist[~q_polyline_mask] = 1e6
    # Obtain the nearest polyline point for each query point and its distance
    _, flat_index = torch.min(points_to_polyline_dist.view(-1, P*S), dim=1)
  
  query_pidx = flat_index // S
  query_sidx = flat_index % S
  query_qidx = torch.arange(len(query_pidx), device=device)

  # For each of the Q points, obtain the distance from nearest polyline points (M)
  query_sidx_left = query_sidx-M # (Q)
  query_sidx_right = query_sidx+M # (Q)

  left_overflow_mask = query_sidx_left < 0
  query_sidx_left[left_overflow_mask] = 0
  query_sidx_right[left_overflow_mask] = 2*M

  query_polyline_len = q_polyline_mask[query_qidx, query_pidx].sum(dim=-1) # (Q)
  right_overflow_mask = query_sidx_right > query_polyline_len - 1 # (Q)
  query_sidx_right[right_overflow_mask] = query_polyline_len[right_overflow_mask] - 1
  query_sidx_left[right_overflow_mask] = query_sidx_right[right_overflow_mask] - 2*M
  assert torch.all((query_sidx_right - query_sidx_left) == 2*M)

  # Create the mask for selecting the correct indices
  max_range = torch.arange(S, device=device)[None, :].repeat(Q, 1)
  mask = (max_range >= query_sidx_left[:, None]) & (max_range < query_sidx_right[:, None])

  query_s_indices = max_range[mask].reshape(Q, 2*M) # (Q, 2*M)
  query_q_indices = query_qidx[:, None].repeat(1, 2*M) # (Q, 2*M)
  query_p_indices = query_pidx[:, None].repeat(1, 2*M) # (Q, 2*M)

  # Obtain the local polyline for each query point
  query_local_polylines = b_polyline_for_dist[query_q_indices, query_p_indices, query_s_indices] # (Q, 2*M, 2)

  assert q_polyline_mask[query_q_indices, query_p_indices, query_s_indices].all()

  # For each of the Q points, compute its signed distance to the local polyline
  query_polyline_signed_dist = _compute_signed_distance_to_polyline_batch(query_points[:, None], query_local_polylines).squeeze(-1) # (Q)

  agent_polyline_signed_dist = query_polyline_signed_dist.reshape(A, T+1, 4) # (A, T+1, 4)

  # reduce to most off-road corner
  agent_polyline_signed_dist = agent_polyline_signed_dist.max(dim=-1)[0] # (A, T+1)

  return agent_polyline_signed_dist, valid_mask

def compute_rollout_offroad_loss_batch(batch, pred_rollout_traj, tgt_rollout_traj, rollout_valid_mask, tgt_mode, T_sample_step=1, M=3):
  # Input:
  # pred_rollout_traj: (B, N, T, 3)
  # tgt_rollout_traj: (B, N, T, 3)
  # rollout_valid_mask: (B, N, T, 3)

  # Compute the offroad loss for every valid agent in the batch
  # pred_agent_offroad_dist: (A, T+1)
  # agent_valid_mask: (B, N) - indicate the agent used to compute the offroad loss
  pred_agent_offroad_dist, agent_valid_mask = compute_rollout_offroad_dist_batch(batch, pred_rollout_traj, T_sample_step=T_sample_step, M=M) # (A, T+1)
  
  # select A_valid agents from A agents that are initially on road; ignore agents that are off-road from the start
  init_onroad_mask = pred_agent_offroad_dist[:, 0] < 0.0 # (A), remains A_valid agents

  pred_dist_matrix = pred_agent_offroad_dist[init_onroad_mask, 1:] # (A_valid, T)
  T = pred_dist_matrix.shape[1]

  if tgt_mode =='any':
    # penalize all the remaining agent being offroad
    pred_loss_matrix = torch.clamp(pred_dist_matrix, min=0.0) # (A_valid, T)
    loss_valid_mask = torch.ones_like(pred_loss_matrix, dtype=torch.bool) # (A_valid, T)
  else:
    with torch.no_grad():
      tgt_agent_offroad_dist, _ = compute_rollout_offroad_dist_batch(batch, tgt_rollout_traj, T_sample_step=T_sample_step, M=M) # (A, T+1)

    tgt_dist_matrix = tgt_agent_offroad_dist[init_onroad_mask, 1:] # (A_valid, T)

    # mask out the invalid tgt rollout steps
    tgt_rollout_valid_mask = rollout_valid_mask[agent_valid_mask][init_onroad_mask].all(dim=-1) # (A_valid, T)
    tgt_rollout_valid_mask = tgt_rollout_valid_mask[:, ::T_sample_step]
    tgt_dist_matrix = tgt_dist_matrix.masked_fill(~tgt_rollout_valid_mask, 0.0)
    
    tgt_offroad_mask = tgt_dist_matrix > OFFROAD_DISTANCE_THRESHOLD # (A_valid, T)

    if tgt_mode == 'temporal_mask':
      # mask out the offroad loss for the time steps where tgt_rollout is offroad
      pred_loss_matrix = pred_dist_matrix.masked_fill(tgt_offroad_mask, 0.0) # (A_valid, T)
      pred_loss_matrix = torch.clamp(pred_dist_matrix, min=0.0) # (A_valid, T)
      loss_valid_mask = ~tgt_offroad_mask
    
    elif tgt_mode == 'agent_mask':
      # mask out the agent that is offroad at any time step in the tgt_rollout
      tgt_offroad_mask_agent = tgt_offroad_mask.any(dim=-1) # (A_valid)
      pred_loss_matrix = pred_dist_matrix.masked_fill(tgt_offroad_mask_agent[:, None], 0.0) # (A_valid, T)
      pred_loss_matrix = torch.clamp(pred_dist_matrix, min=0.0) # (A_valid, T)
      loss_valid_mask = ~tgt_offroad_mask_agent[:, None].repeat(1, T) # (A_valid, T)
    
    elif tgt_mode == 'mse_offroad':
      # compute the mse of offroad distance between pred and tgt for agents that are offroad
      pred_dist_matrix = torch.clamp(pred_dist_matrix, min=0.0) # (A_valid, T)
      tgt_dist_matrix = torch.clamp(tgt_dist_matrix, min=0.0) # (A_valid, T)

      pred_loss_matrix = (pred_dist_matrix - tgt_dist_matrix) ** 2 # (A_valid, T)
      pred_loss_matrix[~tgt_rollout_valid_mask] = 0.0

    elif tgt_mode == 'mse_all':
      pred_loss_matrix = (pred_dist_matrix - tgt_dist_matrix) ** 2 # (A_valid, T)
      pred_loss_matrix[~tgt_rollout_valid_mask] = 0.0 # (A_valid, T)

      loss_valid_mask = tgt_rollout_valid_mask
    
    elif tgt_mode == 'l1':
      pred_loss_matrix = torch.abs(pred_dist_matrix - tgt_dist_matrix) # (A_valid, T)
      pred_loss_matrix[~tgt_rollout_valid_mask] = 0.0 # (A_valid, T)

      loss_valid_mask = tgt_rollout_valid_mask
    

  if loss_valid_mask.sum().item() > 0:
    loss = pred_loss_matrix[loss_valid_mask].mean()
  else:
    loss = torch.tensor(0.0)

  return loss

def _get_edge_info_pytorch(polygon_points):
    """Computes properties about the edges of a polygon using PyTorch.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
          shape (num_polygons, num_points_per_polygon, 2).

    Returns:
        tangent_unit_vectors, normal_unit_vectors, edge_lengths.
    
    Convert from waymo_open_dataset.utils.geometry_utils
    """
    # Shift the polygon points by 1 position to get the edges.
    first_point_in_polygon = polygon_points[:, 0:1, :]
    shifted_polygon_points = torch.cat(
        [polygon_points[:, 1:, :], first_point_in_polygon], dim=1)
    edge_vectors = shifted_polygon_points - polygon_points

    edge_lengths = torch.norm(edge_vectors, dim=2)
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    normal_unit_vectors = torch.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1)

    return tangent_unit_vectors, normal_unit_vectors, edge_lengths

def signed_distance_from_point_to_convex_polygon_pytorch(query_points, polygon_points):
    """Finds the signed distances from query points to convex polygons using PyTorch.

    Args:
        query_points: (batch_size, 2).
        polygon_points: (batch_size, num_points_per_polygon, 2).

    Returns:
        A tensor containing the signed distances of the query points to the
        polygons. Shape: (batch_size,).
    
    Convert from waymo_open_dataset.utils.geometry_utils
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = (
        _get_edge_info_pytorch(polygon_points))

    query_points = query_points.unsqueeze(1)
    vertices_to_query_vectors = query_points - polygon_points
    vertices_distances = torch.norm(vertices_to_query_vectors, dim=2)

    edge_signed_perp_distances = torch.sum(
        -normal_unit_vectors * vertices_to_query_vectors, dim=2)

    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=1)

    projection_along_tangent = torch.sum(
        tangent_unit_vectors * vertices_to_query_vectors, dim=2)
    projection_along_tangent_proportion = torch.div(
        projection_along_tangent, edge_lengths)
    is_projection_on_edge = torch.logical_and(
        projection_along_tangent_proportion >= 0.0,
        projection_along_tangent_proportion <= 1.0)

    edge_perp_distances = torch.abs(edge_signed_perp_distances)
    edge_distances = torch.where(is_projection_on_edge, edge_perp_distances, torch.tensor(float('inf')).to(edge_perp_distances.device))

    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=1)
    min_distance = torch.min(edge_and_vertex_distance, dim=1)[0]
    signed_distances = torch.where(is_inside, -min_distance, min_distance)

    return signed_distances

def _get_downmost_edge_in_box_pytorch(box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box using PyTorch.

    Args:
      box: Tensor of shape (num_boxes, num_points_per_box, 2). The last
        dimension contains the x-y coordinates of corners in boxes.

    Returns:
      A tuple of two tensors:
        downmost_vertex_idx: The index of the downmost vertex, which is also the
          index of the downmost edge. Shape: (num_boxes, 1).
        downmost_edge_direction: The tangent unit vector of the downmost edge,
          pointing in the counter-clockwise direction of the box.
          Shape: (num_boxes, 1, 2).
    
    Convert from waymo_open_dataset.utils.geometry_utils
    """
    NUM_VERTICES_IN_BOX = 4  # Assuming boxes are rectangles

    # Find the index of the downmost vertex in the y dimension.
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=1, keepdim=True)

    # Identify the edge start and end vertices.
    # Add an extra dimension to downmost_vertex_idx and repeat it to match the shape
    downmost_vertex_idx_expanded = downmost_vertex_idx.unsqueeze(-1).repeat(1, 1, 2)
    edge_start_vertex = torch.gather(box, 1, downmost_vertex_idx_expanded)

    edge_end_idx = (downmost_vertex_idx + 1) % NUM_VERTICES_IN_BOX
    # Repeat the edge_end_idx similarly
    edge_end_idx_expanded = edge_end_idx.unsqueeze(-1).repeat(1, 1, 2)
    edge_end_vertex = torch.gather(box, 1, edge_end_idx_expanded)

    # Compute the direction of the downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.norm(downmost_edge, dim=2, keepdim=True)
    downmost_edge_direction = downmost_edge / downmost_edge_length

    return downmost_vertex_idx, downmost_edge_direction

def cross_product_2d_pytorch(a, b):
    """Computes the signed magnitude of the cross product of 2D vectors using PyTorch.

    Args:
        a: Tensor with shape (..., 2).
        b: Tensor with the same shape as `a`.

    Returns:
        An (n-1)-rank tensor with the cross products of paired 2D vectors in `a` and `b`.
    
    Convert from waymo_open_dataset.utils.geometry_utils
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def minkowski_sum_of_box_and_box_points_pytorch(box1_points, box2_points):
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy)
    using PyTorch.

    Args:
        box1_points: Tensor of vertices for box 1, with shape:
            (num_boxes, num_points_per_box, 2).
        box2_points: Tensor of vertices for box 2, with shape:
            (num_boxes, num_points_per_box, 2).

    Returns:
        The Minkowski sum of the two boxes, of size (num_boxes,
        num_points_per_box * 2, 2). The points will be stored in
        counter-clockwise order.
    
    Convert from waymo_open_dataset.utils.geometry_utils
    """
    NUM_VERTICES_IN_BOX = 4  # Assuming boxes are rectangles

    device = box1_points.device

    # Hard coded order to pick points from the two boxes.
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64, device=device)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box_pytorch(box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box_pytorch(box2_points)

    # Calculate condition
    condition = cross_product_2d_pytorch(
        downmost_box1_edge_direction.squeeze(1),
        downmost_box2_edge_direction.squeeze(1)
    ) >= 0.0
    condition = condition.unsqueeze(1).repeat(1, 8)

    # Select points from box1 and box2 based on condition
    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    box1_point_order = (box1_point_order + box1_start_idx) % NUM_VERTICES_IN_BOX
    ordered_box1_points = torch.gather(box1_points, 1, box1_point_order.unsqueeze(-1).expand(-1, -1, 2))

    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = (box2_point_order + box2_start_idx) % NUM_VERTICES_IN_BOX
    ordered_box2_points = torch.gather(box2_points, 1, box2_point_order.unsqueeze(-1).expand(-1, -1, 2))

    minkowski_sum = ordered_box1_points + ordered_box2_points

    return minkowski_sum

def compute_rollout_collision_dist_batch(batch, input_rollout_traj, input_rollout_valid_mask, T_sample_step=1, K=3, collision_threshold=0.0, vehicle_only=False):
  # Get the query agent index and positions

  # Check collision distance with 1hz sampling
  T_sample_step = T_sample_step

  # Compute the collision distance for every valid agent in the batch
  rollout_traj = input_rollout_traj[:, :, ::T_sample_step] # (B, N, T, 2)
  B, N, T = rollout_traj.shape[:3]

  # type_valid_mask (B, N): 1 for valid agents, 0 for invalid agents
  agent_types = batch.extras['io_pairs_batch']['agent_type'][:, 0] # (B, N, 1)

  if vehicle_only:
    # only considered vehicle agents
    type_valid_mask = agent_types == 1 # (B, N)
  else:
    # also consider pedstrain and cyclist agents
    type_valid_mask = agent_types > 0 # (B, N)

  # cnt_valid_mask (B, N): only consider batch with more than 1 agent
  batch_agent_cnt = type_valid_mask.sum(dim=1) # (B)
  cnt_valid_mask = (batch_agent_cnt > 1)[:, None].repeat(1, N) # (B, N)

  valid_mask = type_valid_mask & cnt_valid_mask # (B, N)


  A = valid_mask.sum().item() # A is the number of valid agents in the batch

  device = valid_mask.device
  agent_bidx = torch.arange(B, device=device)[:, None].repeat(1, N)[valid_mask] # (A, 1)
  agent_nidx = torch.arange(N, device=device)[None].repeat(B, 1)[valid_mask] # (A, 1)
  
  # use this mask to determine if the rollout step is valid for each agent
  agent_T_valid_mask = input_rollout_valid_mask.all(dim=-1)[valid_mask] # (A, T)
  
  # use this mask to determine if the agent is a pedestrian
  agent_ped_mask = agent_types[valid_mask] == 2 # (A, 1)


  traj_local = rollout_traj[valid_mask] # (A, T, 3)

  init_xy = batch.extras['io_pairs_batch']['position'][:, 0].unsqueeze(2)[valid_mask] # (A, 1, 2)
  init_h = batch.extras['io_pairs_batch']['heading'][:, 0, :, 0].unsqueeze(2)[valid_mask] # (A, 1)
  sizes = batch.extras['io_pairs_batch']['extend'][:, 0][valid_mask] # (A, 2)

  traj_xy_global = batch_rotate_2D(traj_local[..., :2], init_h) + init_xy # (A, T, 2)
  traj_h_global = wrap_angle(traj_local[:, :, -1] + init_h) # (A, T)
  traj_sizes = sizes[:, None].repeat(1, T, 1) # (A, T, 2)

  # For each agent at each timestamp, compute its distance to other agents in the same batch at the same timestamp
  point_bidx = agent_bidx[:, None].repeat(1, T) # (A, T)
  point_nidx = agent_nidx[:, None].repeat(1, T) # (A, T)
  point_aidx = torch.arange(A, device=device)[:, None].repeat(1, T) # (A, T)
  point_tidx = torch.arange(T, device=device)[None].repeat(A, 1) # (A, T)

  # Collect the indices of valid agents in the same batch at the same timestamp: (A, T, N)
  neigh_bidx = point_bidx[:, :, None].repeat(1, 1, N) # (A, T, N)
  neigh_tidx = point_tidx[:, :, None].repeat(1, 1, N) # (A, T, N)

  # Obtain neighbor's aidx to compute nearest neighbors - in the same batch at the same timestamp
  neigh_aidx = torch.arange(N, device=device)[None, None].repeat(A, T, 1) # (A, T, N)
  neigh_agent_cnt = batch_agent_cnt[point_bidx[:, :, None].repeat(1, 1, N)] # (A, T, N)

  batch_valid_mask = valid_mask.any(-1) # (B)
  # for batch_agent_cnt, mask out agent_cnt from invalid batch (otherwise adix will overflow)
  batch_start_idx = batch_agent_cnt.masked_fill(~batch_valid_mask, 0).cumsum(dim=0)[:-1] # (B-1]
  batch_start_idx = torch.cat([torch.tensor([0], device=device), batch_start_idx]) # (B)
  neigh_start_idx = batch_start_idx[point_bidx][:, :, None].repeat(1, 1, N) # (A, T, N)

  # mask out placeholder agents
  neigh_cnt_mask = neigh_aidx < neigh_agent_cnt # (A, T, N)
  neigh_aidx += neigh_start_idx # (A, T, N)
  neigh_aidx[~neigh_cnt_mask] = -1 # (A, T, N)
  # mask out invalid rollout agent steps
  neigh_T_valid_mask = agent_T_valid_mask[neigh_aidx, neigh_tidx] # (A, T, N)

  # mask out pedestrain-pedestrain pairs
  neigh_ped_mask = agent_ped_mask[neigh_aidx] # (A, T, N)
  point_ped_mask = agent_ped_mask[point_aidx][:, :, None].repeat(1, 1, N) # (A, T, N)
  neigh_ped_ped_mask = neigh_ped_mask & point_ped_mask # (A, T, N)

  # mask out self pairs
  self_mask = point_aidx[:, :, None] == neigh_aidx # (A, T, N)

  neigh_valid_mask = neigh_cnt_mask & neigh_T_valid_mask & (~neigh_ped_ped_mask) & (~self_mask) # (A, T, N)
  neigh_aidx[~neigh_valid_mask] = -1 # (A, T, N)

  with torch.no_grad():
    # Compute distance between traj_xy_global (A, T, 2) and traj_neighbor (A, T, N, 2)
    traj_neighbor = traj_xy_global[neigh_aidx, neigh_tidx] # (A, T, N, 2)

    neigh_dist = (traj_xy_global[:, :, None] - traj_neighbor) ** 2 # (A, T, N, 2)
    neigh_dist = neigh_dist.sum(-1) # (A, T, N)
    neigh_dist[~neigh_valid_mask] = 1e6 # set invalid distance to a large number

    K = min(K, N)

    neigh_topk_idx = torch.topk(neigh_dist, k=K, largest=False, sorted=True).indices # (A, T, K)
    
    neigh_topk_adix = torch.gather(neigh_aidx, 2, neigh_topk_idx) # (A, T, K)
    neigh_topk_tidx = torch.gather(neigh_tidx, 2, neigh_topk_idx) # (A, T, K)

    # for each item, if adix is -1, replace it with 0
    neigh_invalid_mask_k = neigh_topk_adix == -1 # (A, T, K)


    # avoid using negative index for invalid neighbors
    neigh_topk_adix[neigh_invalid_mask_k] = 0 # (A, T, K)

  # Get bbox corners for each agent at each timestamp # (A, T, 4, 2)
  box_input = torch.zeros(A, T, 7, device=device) # (A, T, 7)
  box_input[:, :, :2] = traj_xy_global
  box_input[:, :, -1] = traj_h_global 
  box_input[:, :, 3:5] = traj_sizes

  box_input = box_input.reshape(A*T, 7) # (A*T, 7)
  eval_corners = get_upright_3d_box_corners(box_input)[:, :4, :2]
  eval_corners = eval_corners.reshape(A, T, 4, 2) # (A, T, 4, 2)

  # Get bbox corners for each top-k neighbor at each timestamp # (A, T, K, 4, 2)
  neigh_corners = eval_corners[neigh_topk_adix, neigh_topk_tidx] # (A, T, K, 4, 2)
  eval_corners = eval_corners[:, :, None].repeat(1, 1, K, 1, 1) # (A, T, K, 4, 2)

  eval_corners_flat = eval_corners.reshape(A*T*K, 4, 2) # (A*T*K, 4, 2)
  neigh_corners_flat = neigh_corners.reshape(A*T*K, 4, 2) # (A*T*K, 4, 2)

  # The signed distance between two polygons A and B is equal to the distance between the origin and the Minkowski sum A + (-B), where we generate -B by a reflection
  neg_neigh_corners_flat = -1.0 * neigh_corners_flat
  minkowski_sum = minkowski_sum_of_box_and_box_points_pytorch(eval_corners_flat, neg_neigh_corners_flat) # (A*T*K, 8, 2)
  signed_distances_flat = signed_distance_from_point_to_convex_polygon_pytorch(torch.zeros_like(minkowski_sum[:, 0, :]), minkowski_sum) # (A*T*K)

  # If the two convex shapes intersect, the Minkowski subtraction polygon will contain the origin.
  eval_signed_distances = signed_distances_flat.reshape(A, T, K) # (A, T, K)

  # Mask out agents that do not have any valid neighbors
  eval_signed_distances[neigh_invalid_mask_k] = 1e6 # (A, T, K)

  # Reduce to the minimum signed distance for each agent at each timestamp
  min_signed_distances = eval_signed_distances.min(dim=-1).values # (A, T)

  # collision dist: larger value for more bbox overlap
  collision_dist = -min_signed_distances # (A, T)
  ignore_mask = collision_dist < collision_threshold # (A, T)
  collision_dist[ignore_mask] = 0.0 # (A, T)

  return collision_dist, valid_mask
  # return collision_dist, valid_mask, eval_signed_distances, neigh_topk_adix, eval_corners

def compute_rollout_collision_loss_batch(batch, pred_rollout_traj, tgt_rollout_traj, rollout_valid_mask, tgt_mode, T_sample_step=1, K=3, collision_threshold=0.0, vehicle_only=False):
  # Input:
  # pred_rollout_traj: (B, N, T, 3)
  # rollout_valid_mask: (B, N, T)

  # Compute the collision loss for every valid agent in the batch
  # pred_collision_dit: (A, T) - larger value for more bbox overlap, 0.0 for no collision
  # agent_valid_mask: (B, N) - indicate the agent used to compute the offroad loss

  device = rollout_valid_mask.device

  if tgt_mode == 'any':
    pred_rollout_mask = torch.ones_like(rollout_valid_mask, device=device)
  else:
    pred_rollout_mask = rollout_valid_mask

  pred_collision_dit, agent_valid_mask = compute_rollout_collision_dist_batch(batch, pred_rollout_traj, pred_rollout_mask, T_sample_step=T_sample_step, K=K, collision_threshold=collision_threshold, vehicle_only=vehicle_only)
  
  pred_collision_dit = torch.clamp(pred_collision_dit, min=0.0)
  loss_valid_mask = pred_rollout_mask.all(-1)[agent_valid_mask] # (A, T)

  if tgt_mode == 'any':
    pred_loss_matrix = pred_collision_dit

  else:
    with torch.no_grad():
      tgt_collision_dist, _ = compute_rollout_collision_dist_batch(batch, tgt_rollout_traj, rollout_valid_mask, T_sample_step=T_sample_step, K=K, collision_threshold=collision_threshold, vehicle_only=vehicle_only)

    tgt_collision_dist = torch.clamp(tgt_collision_dist, min=0.0) # (A, T)
    tgt_collision_dist = tgt_collision_dist.masked_fill(~loss_valid_mask, 0.0) # (A, T)
    tgt_collision_mask = tgt_collision_dist > collision_threshold # (A, T)

    if tgt_mode == 'temporal_mask':
      # mask out the collision loss for the time steps where tgt_rollout has collision
      pred_loss_matrix = pred_collision_dit.masked_fill(tgt_collision_mask, 0.0)
      loss_valid_mask = loss_valid_mask & ~tgt_collision_mask
    
    elif tgt_mode == 'agent_mask':
      # mask out the agent that has collision at any time step in the tgt_rollout
      tgt_collision_mask_agent = tgt_collision_mask.any(dim=-1) # (A)
      pred_loss_matrix = pred_collision_dit.masked_fill(tgt_collision_mask_agent[:, None], 0.0) # (A, T)
      loss_valid_mask = loss_valid_mask & ~tgt_collision_mask_agent[:, None] # (A, T)
    
    elif tgt_mode == 'l1':
      # compute the mse of collision distance between pred and tgt for agents that have collision
      pred_loss_matrix = torch.abs(pred_collision_dit - tgt_collision_dist)


  if loss_valid_mask.sum().item() > 0:
    loss = pred_loss_matrix[loss_valid_mask].mean()
  else:
    loss = torch.tensor(0.0)

  return loss
  

loss_func_dict = {'ego_mse_k': ego_mse_k_way, 'motion_pred_mse_k': motion_mse_k_way, 'paired_mse_k': paired_mse_k_way, 'empty': empty, 'init_vel_pred': init_vel_pred, 'goal_pred': goal_pred, 'extend_pred': extend_pred, 'pass': pass_loss}