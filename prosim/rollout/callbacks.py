import torch
import random
import tqdm
import time
import wandb
import glob
import imageio
import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from PIL import Image
# import tensorflow as tf

from trajdata.simulation import SimulationScene

from .utils import rollout_scene_loop
from .metrics import CrashDetect, GoalReach

# from .waymo_utils import get_waymo_file_template, get_waymo_scene_object, plot_waymo_gt_trajectory, rollout_states_to_joint_scene, plot_waymo_rollout_trajectory
# from waymo_open_dataset.utils.sim_agents import submission_specs

from .gpu_utils import get_waymo_specification, parallel_rollout_batch, obtain_rollout_trajs_in_world, obtain_waymo_scenario_rollouts, joint_scene_from_rollout

from pathlib import Path
# from waymo_open_dataset.protos import sim_agents_metrics_pb2
# from google.protobuf import text_format
# from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

class rollout_callback(Callback):
  def __init__(self, config, rollout_dataset):
    self.config = config
    self.r_cfg = self.config.ROLLOUT
    self.v_cfg = self.r_cfg.VISUALIZE
    self.p_cfg = self.r_cfg.POLICY
    self.m_cfg = self.r_cfg.METRIC
    self.dataset = rollout_dataset
    
    self.gif_template = '/tmp/rollout_scene_{}.gif'
    self.track_template = '/tmp/rollout_track_{}.jpg'

    self.use_waymo = 'waymo' in config.DATASET.SOURCE.ROLLOUT[0]

    if self.use_waymo:
      self.metric_names = ['goal_reach']
    else:
      self.metric_names = ['crash_detect', 'goal_reach']

    super().__init__()

  def on_test_start(self, trainer, pl_module):
    device_id = trainer.local_rank
    if self.r_cfg.ENABLE:
      print(f"device {device_id} - testing - begin rollout dataset")
      self.rollout_dataset(trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    device_id = trainer.local_rank
    if self.r_cfg.ENABLE:
      ep = trainer.current_epoch
      if ep >= self.r_cfg.WARMUP_EPOCH and ep % self.r_cfg.INTERVAL_EPOCH == 0:
        print(f"device {device_id} - epoch # {ep} end  - begin rollout dataset")
        self.rollout_dataset(trainer, pl_module)

  def rollout_dataset(self, trainer, pl_module):
    scene_cnt = self.dataset.num_scenes()
    device_cnt = trainer.num_devices
    device_id = trainer.local_rank
    prompt_values = self.r_cfg.PROMPT_VALUES

    # uniformly split scenes to different devices
    scene_ids = [i for i in range(scene_cnt) if (i % device_cnt) == device_id]
  
    print(f"device {device_id} - rollout scenes: {scene_ids}")

    metric_results = []

    for scene_id in tqdm.tqdm(scene_ids, desc=f"device {device_id} - rollout scenes"):
      scene = self.dataset.get_scene(scene_id)

      print(f"device {device_id} - rollout scene: {scene.name}")

      if self.v_cfg.ENABLE:
        to_plot_gif =  scene_id % self.v_cfg.GIF_INTERVAL == 0
        to_plot_track = self.use_waymo and (scene_id % self.v_cfg.TRACK_INTERVAL == 0)
      else:
        to_plot_gif = False
        to_plot_track = False

      if prompt_values is None:
        rollout_p_values = [None]
      else:
        rollout_p_values = prompt_values

      # iterate over all prompt values
      for p_value in rollout_p_values:
        metric = self.rollout_scene(scene, pl_module, to_plot_gif, to_plot_track, scene_id, p_value)
        metric_results.append(metric)
    
    if distributed.is_initialized() and distributed.get_world_size() > 1:
      distributed.barrier()
      print(f"device {device_id} - barraier passed: all devices finished rollout scenes")

    if self.v_cfg.ENABLE and trainer.is_global_zero:
      gif_ids = [i for i in range(scene_cnt) if (i % self.v_cfg.GIF_INTERVAL) == 0]
      self._log_gif(trainer, gif_ids)

      self._log_track(trainer)
      
    self._log_metric(trainer, pl_module, metric_results)

  def _process_binary_metric(self, metric_results, mode):
    bin_logs = []
    for metric in metric_results:
      for _, bin_result in metric[mode].items():
        bin_logs.append(bin_result)
    rate = np.mean(bin_logs)
    return rate, len(bin_logs)

  def _log_metric(self, trainer, pl_module, metric_results):
    sync_dist = trainer.num_devices > 1 and self.r_cfg.SYNC_DIST

    for metric_name in self.metric_names:
      rate, agent_cnt = self._process_binary_metric(metric_results, metric_name)
      if agent_cnt > 0:
        pl_module.log(f'rollout_metric/{metric_name}', rate, on_epoch=True, on_step=False, sync_dist=sync_dist, batch_size=agent_cnt)
        print(f"device {trainer.local_rank} - logged {metric_name} rate: {rate} - agent_cnt: {agent_cnt}")
      else:
        print(f"device {trainer.local_rank} - no agent for {metric_name}")

  def _log_gif(self, trainer, vis_scene_ids):
    prompt_values = self.r_cfg.PROMPT_VALUES
    
    for scene_id in vis_scene_ids:
      if prompt_values is None:
        scene_names = [scene_id]
      else:
        scene_names = [f"{scene_id}_prompt_{p_value}" for p_value in prompt_values]

      for scene_name in scene_names:
        save_file = self.gif_template.format(scene_name)
        trainer.logger.experiment.log(
          {f"rollout_video/scene_{scene_name}": [wandb.Video(save_file, format='gif')],},
        )
    
  def _log_track(self, trainer):
    track_files = glob.glob(self.track_template.format('*'))

    for track_file in track_files:
      scene_id = track_file.split('/')[-1].split('.')[0].split('_')[-1]
      trainer.logger.experiment.log(
        {f"rollout_track/scene_{scene_id}": [wandb.Image(track_file)],},
      )

  def rollout_scene(self, scene, pl_module, to_plot_gif, to_plot_track, scene_id, prompt_value):
    start_frame = self.p_cfg.POLICY_START_FRAME
    sim_scene: SimulationScene = SimulationScene(
      env_name=self.config.DATASET.SOURCE.ROLLOUT,
      scene_name=f"sim_scene",
      scene=scene,
      dataset=self.dataset,
      init_timestep=start_frame,
      freeze_agents=True,
    )
    max_step = min(start_frame+self.p_cfg.MAX_STEPS, sim_scene.scene.length_timesteps-1)

    scene_agent_names = [agent.name for agent in scene.agents]

    ego_agent_id = None
    center_agent = self.r_cfg.CENTER_AGENT
    ego_obs = sim_scene.get_obs(agent_names=[center_agent])
    all_neigh_names = ego_obs.neigh_names[0]
    random.shuffle(all_neigh_names)

    if self.use_waymo:
      control_agents = ['ego']
      scene_template = get_waymo_file_template(self.config)
      waymo_scene = get_waymo_scene_object(scene.name, scene_template)
      sim_agent_ids = submission_specs.get_sim_agent_ids(waymo_scene)

      for sim_agent_id in sim_agent_ids:
        if str(sim_agent_id) not in scene_agent_names:
          ego_agent_id = sim_agent_id
        elif str(sim_agent_id) in all_neigh_names:
          control_agents.append(str(sim_agent_id))
    else:
      # get all the neighbor agents that appear in the first frame
      control_agents = [center_agent] + all_neigh_names

    control_agents = control_agents[:self.r_cfg.CONTROL_NUM]

    sim_scene, vis_frames, rollout_global_states = rollout_scene_loop(sim_scene, pl_module, control_agents, start_frame, max_step, center_agent, self.config, to_plot_gif, prompt_value)

    goal_metric = GoalReach(control_agents, self.m_cfg.ROLLOUT_TRAJ.DIST_THRESHOLD)
    if 'crash_detect' in self.metric_names:
      agent_extends = {}
      for agent in scene.agents:
        ext = agent.extent
        agent_extends[agent.name] = [ext.length, ext.width]
      crash_metric = CrashDetect(control_agents, agent_extends, self.m_cfg.CRASH_RATE.IOU_THRESHOLD, mode='sim')
      metric_results = sim_scene.get_metrics([crash_metric, goal_metric])
    else:
      metric_results = sim_scene.get_metrics([goal_metric])

    if to_plot_gif:
      if prompt_value is None:
        scene_name = scene_id
      else:
        scene_name = f"{scene_id}_prompt_{prompt_value}"

      save_file = self.gif_template.format(scene_name)
      duration = 1000 / self.v_cfg.FPS
      imageio.mimsave(save_file, vis_frames, duration=duration, loop=0)
    
    if to_plot_track:
      gt_track_plot = plot_waymo_gt_trajectory(waymo_scene, start_frame, max_step)
      joint_scene = rollout_states_to_joint_scene(waymo_scene, rollout_global_states, control_agents, ego_agent_id, start_frame)
      rollout_track_plot = plot_waymo_rollout_trajectory(waymo_scene, joint_scene)

      track_img = Image.fromarray(np.concatenate([gt_track_plot, rollout_track_plot], axis=1))
      track_img.save(self.track_template.format(scene_name))

    return metric_results

class rollout_callback_gpu(rollout_callback):
  def __init__(self, config, rollout_dataset):
    super().__init__(config, rollout_dataset)
    self.M = config.ROLLOUT.PARALLEL_NUM

    metric_path = Path(metrics.__file__).parent
    config_path = metric_path / 'challenge_config.textproto'
    with open(config_path, 'r') as f:
        waymo_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), waymo_config)
    
    self.waymo_config = waymo_config

  def _get_subsampled_dataloader(self, trainer):
    scene_cnt = self.dataset.num_scenes()
    device_cnt = trainer.num_devices
    device_id = trainer.local_rank

    indices = [i for i in range(scene_cnt) if (i % device_cnt) == device_id]
    local_dataset = Subset(self.dataset, indices)

    print(f"device {device_id} - rollout scenes: {indices}")

    data_loader = DataLoader(local_dataset, batch_size=1, shuffle=False, num_workers=self.r_cfg.NUM_WORKERS, collate_fn=self.dataset.get_collate_fn())
    
    return data_loader

  def _plot_waymo_track(self, waymo_scene, joint_scenes, scene_name):
      gt_track_plot = plot_waymo_gt_trajectory(waymo_scene, 10, 90)
      rollout_track_plots = [plot_waymo_rollout_trajectory(waymo_scene, joint_scene) for joint_scene in joint_scenes]
      track_img = Image.fromarray(np.concatenate([gt_track_plot] + rollout_track_plots, axis=1))
      track_img.save(self.track_template.format(scene_name))

  def _log_metric(self, trainer, pl_module, metric_results):
    sync_dist = trainer.num_devices > 1 and self.r_cfg.SYNC_DIST

    for line in str(metric_results).split('\n'):
      if ':' not in line or 'scenario_id' in line:
        continue

      key, value = line.split(':')
      key = key.strip()
      value = float(value.strip())

      pl_module.log(f'rollout_metric/{key}', value, on_epoch=True, on_step=False, sync_dist=sync_dist, batch_size=1)

  def rollout_dataset(self, trainer, pl_module):
    data_loader = self._get_subsampled_dataloader(trainer)

    for idx, batch in enumerate(tqdm.tqdm(data_loader, desc=f"device {trainer.local_rank} - rollout scenes")):
      start_rollout = time.time()
      print(f"device {trainer.local_rank} - begin scene: {batch.scene_ids[0]}")

      batch, waymo_scene, ego_sim_agent_id = get_waymo_specification(batch, self.config)
      batch.to(pl_module.device)
      
      with torch.no_grad():
        result_M = parallel_rollout_batch(batch, self.M, pl_module)
        rollout_trajs_in_world_M, object_ids_M = obtain_rollout_trajs_in_world(batch, result_M)

      end_rollout = time.time()

      print(f"device {trainer.local_rank} - finish scene: {batch.scene_ids[0]} - time: {end_rollout - start_rollout}")

      # scenario_rollouts = obtain_waymo_scenario_rollouts(waymo_scene, rollout_trajs_in_world_M, object_ids_M, ego_sim_agent_id)

      vis_joint_scenes = [joint_scene_from_rollout(waymo_scene, rollout_trajs_in_world_M[i], object_ids_M[i], ego_sim_agent_id) for i in range(self.M)]

      self._plot_waymo_track(waymo_scene, vis_joint_scenes, batch.scene_ids[0])

      scene_name = batch.scene_ids[0]
      scene_id = int(scene_name.split('_')[-1])

    if distributed.is_initialized() and distributed.get_world_size() > 1:
      distributed.barrier()
      print(f"device {trainer.local_rank} - barraier passed: all devices finished rollout scenes")

    if self.v_cfg.ENABLE and trainer.is_global_zero:
      self._log_track(trainer)

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
class rollout_callback_distributed(rollout_callback_gpu):
  def __init__(self, config, rollout_dataset, save_dir, wandb_id):
    self.config = config
    self.r_cfg = self.config.ROLLOUT
    self.request_path = self.config.ROLLOUT_REQUEST_PATH
    os.makedirs(self.request_path, exist_ok=True)

    self.time_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    self.save_dir = Path(save_dir)
    self.wandb_id = wandb_id

    self.scene_cnt = rollout_dataset.num_scenes()

    super().__init__(config, rollout_dataset)

    # for distributed, we only visualize rollout a single version for show.
    self.M = self.r_cfg.ONLINE_M

  def on_test_start(self, trainer, pl_module):
    if self.r_cfg.ENABLE:
      print(f"testing - submit rollout dataset request")
      self._start_rollout(trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    if self.r_cfg.ENABLE:
      ep = trainer.current_epoch
      if ep >= self.r_cfg.WARMUP_EPOCH and ep % self.r_cfg.INTERVAL_EPOCH == 0:
        print(f"epoch # {ep} end  - submit rollout dataset request")
        self._start_rollout(trainer, pl_module)
  
  def _start_rollout(self, trainer, pl_module):
    print(f"device {trainer.local_rank} - begin rollout dataset")
    
    if trainer.is_global_zero and self.r_cfg.REQUEST_METRIC:
      self.submit_rollout_request(trainer, pl_module)
    
    if self.v_cfg.ENABLE:
      self.rollout_dataset(trainer, pl_module)
    
    print(f"device {trainer.local_rank} - finish rollout dataset")

  def _get_subsampled_dataloader(self, trainer):
    scene_cnt = self.dataset.num_scenes()
    device_cnt = trainer.num_devices
    device_id = trainer.local_rank

    indices = [i for i in range(scene_cnt) if (i % device_cnt) == device_id]
    vis_num = self.v_cfg.DISTRIBUTED_VIS_NUM
    indices = indices[:vis_num]

    local_dataset = Subset(self.dataset, indices)

    print(f"device {device_id} - rollout scenes: {indices}")

    data_loader = DataLoader(local_dataset, batch_size=1, shuffle=False, num_workers=self.r_cfg.NUM_WORKERS, collate_fn=self.dataset.get_collate_fn())
    
    return data_loader

  def submit_rollout_request(self, trainer, pl_module):
    ep = trainer.current_epoch

    rollout_cpkt_path = self.save_dir / f"rollout_ep_{ep}.ckpt"
    trainer.save_checkpoint(rollout_cpkt_path)

    # write the request file
    exp_name = os.path.join(self.config.EXPERIMENT_DIR, self.config.EXPERIMENT_NAME)
    exp_name = exp_name.replace('/', '_')
    exp_name = exp_name.replace('results_', '')
    
    request_file = f"{exp_name}_{self.time_str}_epoch_{ep}.json"
    request_file = os.path.join(self.request_path, request_file)

    request = {
      "ckpt_path": str(rollout_cpkt_path),
      "exp_folder": str(self.save_dir),
      "time_str": self.time_str,
      "epoch": ep,
      "global_step": trainer.global_step,
      "wandb_id": self.wandb_id,
      "num_scenes": self.scene_cnt,
    }

    with open(request_file, 'w') as f:
      json.dump(request, f)
    
    print(f"submitted rollout request: {request_file}")