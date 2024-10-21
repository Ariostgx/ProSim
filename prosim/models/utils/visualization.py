import torch
import random
from typing import Any, Optional
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

from trajdata.utils.state_utils import StateTensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import Callback

from trajdata.visualization.vis import plot_agent_batch, plot_scene_batch
from trajdata.utils.arr_utils import transform_coords_np
from prosim.dataset.data_utils import rotate, default_trajdata_cfg
from prosim.loss.loss_func import pair_names_to_indices


def get_heatmap(x, y, prob, s, bins=1000):
  heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=prob, density=True)

  heatmap = gaussian_filter(heatmap, sigma=s)

  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
  return heatmap.T, extent


def vis_agent_traj_pred(batch, output, idx):
  fig, ax = plt.subplots()

  # only visualize the first batch element

  ax = plot_agent_batch(batch, ax=ax, batch_idx=idx, legend=True, show=False, close=False)

  ego_motion_pred = output['motion_pred'][0].detach().cpu().numpy()
  ego_motion_prob = output['motion_prob'][0].detach().cpu().numpy()

  K = ego_motion_prob.shape[0]

  prob_index = np.argsort(-ego_motion_prob)
  ego_motion_pred = ego_motion_pred[prob_index]

  ax.scatter(ego_motion_pred[0, :, 0], ego_motion_pred[0, :, 1], s=15.0, c='r')

  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_top = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  if K == 1:
    return [image_top]

  colors = plt.cm.tab10.colors
  color_select = colors[:2] + colors[4:]

  for i in range(K-1):
    color = color_select[(i+1) % len(color_select)]
    ax.scatter(ego_motion_pred[1+i, :, 0], ego_motion_pred[1+i, :, 1], s=10.0, color=color)
  
  ax.legend()
  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_K = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  plt.close('all')

  return [image_top, image_K]

def vis_scene_goal_pred_heatmap(batch, output, config, agent_name):
  batch.to('cpu')

  # draw a heatmap of the goal prediction for the first agent
  if config.MODEL.DECODER.GOAL_PRED.HEATMAP_PRED:
    agent_names = [output['pair_names'][i].split('-')[1] for i in range(len(output['pair_names']))]
    vis_idx = agent_names.index(agent_name)

    map_xy = batch.extras['init_map']['input'][0][..., :2].detach().cpu()
    map_prob = output['goal'][vis_idx].detach().cpu().softmax(dim=-1)

    if len(map_xy.shape) == 3:
      map_position = batch.extras['init_map']['position'][0].cpu()
      map_heading = batch.extras['init_map']['heading'][0].cpu()

      map_xy = rotate(map_xy[..., 0], map_xy[..., 1], map_heading)
      map_xy += map_position
      map_xy = map_xy.numpy()

      N = map_xy.shape[1]

      map_prob = map_prob[:, None].expand(-1, N).reshape(-1)
      map_xy = map_xy.reshape(-1, 2)
      
      map_mask = batch.extras['init_map']['input'][0][..., 4].cpu() > 0
      map_mask = map_mask.reshape(-1)
      
      s = 5
    
    else:
      map_xy = map_xy.numpy()
      map_mask = batch.extras['init_map']['mask'][0].cpu()
    
      s = 20

    heatmap = get_heatmap(map_xy[:, 0][map_mask], map_xy[:, 1][map_mask], map_prob[map_mask], s, bins=1000)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax = plot_scene_batch(batch, ax=ax, batch_idx=0, legend=False, show=False, close=False, controlled_names=[agent_name])

    ax.imshow(heatmap[0], extent=heatmap[1], origin='lower', cmap=cm.jet, alpha=0.5)

    fig.canvas.draw()
    image_heatmap = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_heatmap = image_heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return image_heatmap


def vis_scene_traj_pred(batch, output, config, target_T = 0):
  default_obs_format = default_trajdata_cfg['obs_format']

  has_motion_pred = 'motion_pred' in output
  
  fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

  batch.to('cpu')

  # only visualize the first batch element at the first time step
  pred_step = config.DATASET.FORMAT.TARGET.STEPS
  hist_step = config.DATASET.FORMAT.HISTORY.STEPS
  
  ori_hist = batch.agent_hist.clone()
  ori_fut = batch.agent_fut.clone()

  full_traj = StateTensor.from_array(torch.cat([batch.agent_hist, batch.agent_fut], dim=2), format=default_obs_format)
  batch.agent_hist = full_traj[:, :, :hist_step+target_T]
  # batch.agent_fut = full_traj[:, :, hist_step+target_T:hist_step+target_T+pred_step]

  aug_agents = []

  ax = plot_scene_batch(batch, ax=ax, batch_idx=0, legend=False, show=False, close=False, controlled_names=aug_agents)
  
  batch.agent_hist = ori_hist
  batch.agent_fut = ori_fut

  vis_indices = []
  for idx, name in enumerate(output['pair_names']):
    batch_id, _, T = name.split('-')
    if batch_id == '0' and int(T) == target_T:
      vis_indices.append(idx)

  batch_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][0]

  for vidx in vis_indices:
    name = output['pair_names'][vidx]

    rollout_id = '-'.join(name.split('-')[:2])
    roll_traj = output['rollout_trajs'][rollout_id]['traj'].cpu().detach().numpy()
    init_pos = output['rollout_trajs'][rollout_id]['init_pos'].cpu().detach().numpy()
    init_heading = output['rollout_trajs'][rollout_id]['init_heading'].cpu().detach().numpy()

    roll_traj = rotate(roll_traj[..., 0], roll_traj[..., 1], init_heading) + init_pos
    ax.scatter(roll_traj[:, 0], roll_traj[:, 1], color='r', s=10, marker='x')

    hasOneText = False
    for key in batch.extras['condition'].keys():
      if 'OneText' in key:
        hasOneText = True
        break

    # if hasOneText:
    nidx = batch_agent_names.index(name.split('-')[1])
    agent_name = f'A{nidx}'
    # else:
    #   agent_name = name.split('-')[1][:5]
    
    ax.text(init_pos[0], init_pos[1]-1.5, agent_name, fontsize=8, color='blue', ha='center', va='bottom', fontweight=300)

    nidx = batch_agent_names.index(name.split('-')[1])

    if 'reconst_pred' in output:
      reconst_pred = output['reconst_pred'][vidx].detach().cpu().numpy()
      reconst_pred = rotate(reconst_pred[0], reconst_pred[1], init_heading)[0] + init_pos
      
      ax.scatter(reconst_pred[0], reconst_pred[1], color='r', s=50, marker='x')
      ax.text(reconst_pred[0], reconst_pred[1]+2.0, agent_name, fontsize=8, color='r', ha='center', va='top', fontweight=100)
    
    if 'goal_point' in output:
      goal_pos = output['goal_point'][vidx].detach().cpu().numpy() # [K, 2]
      goal_dist = output['goal_prob'][vidx].detach().cpu().numpy() # [K]

      goal_rank = np.argsort(-goal_dist) # [K]
      if 'ego' in name:
        vis_n = len(goal_rank) // 2
      else:
        # vis_n = min(2, len(goal_rank))
        vis_n = 0

      for k in range(vis_n):
        goal = goal_pos[goal_rank[k]]
        goal = rotate(goal[0], goal[1], init_heading)[0] + init_pos
        ax.scatter(goal[0], goal[1], color='r', s=50, marker='+')
        ax.text(goal[0], goal[1]+2.0, agent_name, fontsize=8, color='r', ha='center', va='top', fontweight=100)
      
      for k in range(vis_n):
        goal = goal_pos[goal_rank[-k]]
        goal = rotate(goal[0], goal[1], init_heading)[0] + init_pos
        ax.scatter(goal[0], goal[1], color='g', s=50, marker='+')
        ax.text(goal[0], goal[1]+2.0, agent_name, fontsize=8, color='g', ha='center', va='top', fontweight=100)

    if 'goal' in batch.extras['condition'].keys():
      goal_nidx = batch.extras['condition']['goal']['prompt_idx'][0, :, 0].cpu().numpy().tolist()
      if nidx in goal_nidx:
        cidx = goal_nidx.index(nidx)

        goal_pos = batch.extras['condition']['goal']['input'][0, cidx, :2].cpu().numpy()
        goal_pos = rotate(goal_pos[0], goal_pos[1], init_heading)[0] + init_pos

        ax.scatter(goal_pos[0], goal_pos[1], color='g', s=50, marker='x')
        ax.text(goal_pos[0], goal_pos[1]+2.0, agent_name, fontsize=8, color='g', ha='center', va='top', fontweight=100)
  
    if 'drag_point' in batch.extras['condition'].keys():
      drag_point_nidx = batch.extras['condition']['drag_point']['prompt_idx'][0, :, 0].cpu().numpy().tolist()
      if nidx in drag_point_nidx:
        cidx = drag_point_nidx.index(nidx)

        drag_pos = batch.extras['condition']['drag_point']['input'][0, cidx, :, :2].cpu().numpy()
        drag_pos = rotate(drag_pos[..., 0], drag_pos[..., 1], init_heading) + init_pos
        ax.scatter(drag_pos[..., 0], drag_pos[..., 1], color='b', s=20, marker='x')

  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  result = [image]
  plt.close('all')
  
  return result


def vis_rollout_traj_pred(ego_obs, action_dicts, action_idx, global_trajs):
  fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
  fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=100)

  # only visualize the first batch element at the first time step

  ego_obs.to('cpu')

  controlled_names = list(action_dicts.keys())

  ax = plot_agent_batch(ego_obs, ax=ax, batch_idx=0, legend=False, show=False, close=False, with_history=False, with_future=False, controlled_names=controlled_names)
  ax2 = plot_agent_batch(ego_obs, ax=ax2, batch_idx=0, legend=False, show=False, close=False, with_history=False, with_future=False, controlled_names=controlled_names)
  colors = plt.cm.tab10.colors
  color_select = colors[:2] + colors[4:]
  
  K_vis = 3

  center_from_world_tf = ego_obs.agents_from_world_tf[0].cpu().numpy()

  for agent_name, output in action_dicts.items():
    motion_pred = global_trajs[agent_name]['motion_pred'][..., :2]
    motion_pred = motion_pred[:, action_idx:, ...]
    motion_prob = output['motion_prob'].detach().cpu().numpy()
    motion_pred = transform_coords_np(motion_pred, center_from_world_tf)

    top_k_indices = np.argsort(-motion_prob)

    for pred_idx in range(min(K_vis, len(top_k_indices))):
      top_k_idx = top_k_indices[pred_idx]
      traj_pred = motion_pred[top_k_idx]

      if pred_idx == 0:
        color = 'r'
        ax.scatter(traj_pred[:, 0], traj_pred[:, 1], s=10.0, color=color)
      else:
        color = color_select[pred_idx % len(color_select)]
      ax2.scatter(traj_pred[:, 0], traj_pred[:, 1], s=7.0, color=color)
    
    for goal_type in ['policy_goal', 'pred_goal']:
      color = 'g' if goal_type == 'policy_goal' else 'r'
      
      if goal_type in global_trajs[agent_name]:
        goal = global_trajs[agent_name][goal_type]
        goal = transform_coords_np(goal, center_from_world_tf)

        ax.scatter(goal[0], goal[1], s=35.0, color=color, marker='x')
        ax2.scatter(goal[0], goal[1], s=35.0, color=color, marker='x')
      
        
  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  fig2.canvas.draw()
  image2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
  image2 = image2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

  plt.close('all')

  return [image, image2]

class visualization_callback(Callback):
  def __init__(self, config):
    self.config = config
    super().__init__()

  def _shared_step(self, trainer, pl_module, outputs, batch, batch_idx, mode):
    if self.config.ENABLE_VIS and trainer.is_global_zero:
      if batch_idx % self.config.VIS_INTERVAL == 0:
        pl_module._visualize(batch, outputs['model_output'], mode)
    
      if 'inst_enum_outputs' in outputs:
        cfg = pl_module.config
        heatmaps = []
        agent_name = outputs['inst_enum_agent_name']
        enum_outputs = outputs['inst_enum_outputs']
        for enum_output in enum_outputs:
          heatmaps.append(vis_scene_goal_pred_heatmap(batch, enum_output, cfg, agent_name))
        
        pl_module._log_image(heatmaps, mode, 'goal_pred_heatmap')

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
    if dataloader_idx is None or dataloader_idx == 0:
      self._shared_step(trainer, pl_module, outputs, batch, batch_idx, 'val')
    

  def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
    if dataloader_idx is None or dataloader_idx == 0:
      self._shared_step(trainer, pl_module, outputs, batch, batch_idx, 'test')