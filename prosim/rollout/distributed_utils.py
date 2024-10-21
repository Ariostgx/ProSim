import sys
import pickle
import numpy as np
import torch
import json
import glob

import time
import tqdm
import random
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from pathlib import Path
# from waymo_open_dataset.protos import sim_agents_metrics_pb2
# from google.protobuf import text_format

from prosim.config.default import Config, get_config
from prosim.core.registry import registry
from prosim.rollout.gpu_utils import get_waymo_specification, parallel_rollout_batch, obtain_rollout_trajs_in_world, obtain_waymo_scenario_rollouts
# from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

import psutil

# Function to get current GPU memory usage
def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)

def get_ckpt_and_config(cfg_file):
  config = get_config(cfg_file, cluster='slurm')
  result_root = '/lustre/fsw/portfolios/nvr/users/shuhant/results/'
  result_path = os.path.join(result_root, config.EXPERIMENT_DIR, config.EXPERIMENT_NAME)

  print('result_path', result_path)

  hpc_ckpts = sorted(glob.glob(os.path.join(result_path, 'hpc_ckpt_*')))
  if len(hpc_ckpts) > 0:
    # ckpt = hpc_ckpts[0]
    ckpt = hpc_ckpts[-1]
    print('loading from ckpt', ckpt)
  else:
    ckpts = sorted(glob.glob(os.path.join(result_path, 'prosim_mixture', '*', 'checkpoints', 'last.ckpt')))
    # ckpt = ckpts[0]
    if len(ckpts) > 0:
      ckpt = ckpts[-1]
      print('loading from ckpt', ckpt)
    else:
      ckpt = None
      print('no ckpt found!')

  return ckpt, config

def get_model_and_config(cfg_file):
  model_cls = registry.get_model("prosim_policy_relpe_T_step_temporal_close_loop")
  ckpt, config = get_ckpt_and_config(cfg_file)
  model = model_cls.load_from_checkpoint(ckpt, config=config, strict=False, map_location='cpu').cpu()

  return model, config

def print_system_mem_usage():
  # Get the memory details
  memory = psutil.virtual_memory()

  # Total memory
  total_memory = memory.total

  # Available memory
  available_memory = memory.available

  # Used memory
  used_memory = memory.used

  # Memory usage percentage
  memory_percent = memory.percent

  print(f"Total Memory: {total_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Available Memory: {available_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Used Memory: {used_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Memory Usage: {memory_percent}%", flush=True)


def check_mem_usage(pid):
  # PID is the process ID of the process you want to check
  process = psutil.Process(pid)

  # Get memory usage information
  memory_info = process.memory_info()

  data = f"RSS: {memory_info.rss / (1024 * 1024)} MB, Shared: {memory_info.shared / (1024 * 1024)} MB"

  return data

def rollout_scene_distributed(config, M, ckpt_path, rollout_id, save_metric, save_rollout, top_k, traj_noise_std, action_noise_std, sampler_cfg=None, smooth_dist=5.0):
  # print_system_mem_usage()

  dataset = registry.get_dataset("prosim_imitation")(config, 'rollout', centric='scene')
  print(f'{os.getpid()} initialized dataset')
  
  config.ROLLOUT.POLICY['TOP_K'] = top_k
  config.MODEL.POLICY.ACT_DECODER['RANDOM_NOISE_STD'] = action_noise_std
  print('set top_k: ', top_k, flush=True)
  print('config.ROLLOUT.POLICY: ', config.ROLLOUT.POLICY, flush=True)
  print('traj noise std: ', traj_noise_std, flush=True)
  print('action noise std: ', action_noise_std, flush=True)
  
  model_cls = registry.get_model("prosim_policy_relpe_T_step_temporal")
  model = model_cls.load_from_checkpoint(ckpt_path, config=config, map_location='cpu', strict=False)
  print(f'{os.getpid()} initialized model from {ckpt_path}')

  if sampler_cfg is not None and sampler_cfg != 'None':
    sampler_model, _ = get_model_and_config(sampler_cfg)
    print(f'{os.getpid()} initialized sampler model from {sampler_cfg}')
  else:
    sampler_model = None

  # print_system_mem_usage()

  all_scene_idx = list(range(dataset.num_scenes()))
  random.shuffle(all_scene_idx)

  # all_scene_idx = [1338]

  exp_name = config.EXPERIMENT_NAME
  save_root = os.path.join(config.SAVE_DIR, config.EXPERIMENT_DIR, str(exp_name))
  save_root = Path(save_root)

  rollout_root = save_root/ 'rollout' / f'{rollout_id}_{M}_top_{top_k}'

  status_path = rollout_root / 'status'
  status_path.mkdir(parents=True, exist_ok=True)

  assert save_metric or save_rollout, 'save_metric or save_rollout must be True'

  
  metrics_path = rollout_root / 'metrics'
  metrics_path.mkdir(parents=True, exist_ok=True)
  
  rollout_path = rollout_root / 'rollout'
  rollout_path.mkdir(parents=True, exist_ok=True)

  print(f'{os.getpid()} initialized rollout path: {rollout_root}', flush=True)

  metric_path = Path(metrics.__file__).parent
  config_path = metric_path / 'challenge_config.textproto'
  with open(config_path, 'r') as f:
    waymo_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(f.read(), waymo_config)

  for scene_idx in tqdm.tqdm(all_scene_idx, desc=f'rollout on {os.getpid()}'):
    status_file = status_path / f'scene_{scene_idx}.start'

    if status_file.exists():
      print(f'other process already work on scene {scene_idx} ', flush=True)
      continue

    os.system('touch ' + str(status_file))

    rollout_file = rollout_path / f'scene_{scene_idx}.pb'
    metric_file = metrics_path / f'scene_{scene_idx}.json'

    # result_exists = rollout_file.exists()
    if save_rollout and save_metric:
      result_exists = rollout_file.exists() and metric_file.exists()
    elif save_rollout:
      result_exists = rollout_file.exists()
    elif save_metric:
      result_exists = metric_file.exists()
    
    if result_exists:
      print(f'process({os.getpid()}): scene {scene_idx} already exists', flush=True)
      continue

    try:
      # print_system_mem_usage()

      local_dataset = Subset(dataset, [scene_idx])
      data_loader = DataLoader(local_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.get_collate_fn())

      for batch in data_loader:
        break

      # print_system_mem_usage()

      print(f'process({os.getpid()}): start rollout scene {scene_idx} - M = {M}', flush=True)

      start = time.time()
      with torch.no_grad():
          batch, waymo_scene, ego_sim_agent_id = get_waymo_specification(batch, config)
          result_M = parallel_rollout_batch(batch, M, model, top_K=top_k, sampler_model=sampler_model, smooth_dist=smooth_dist)
          rollout_trajs_in_world_M, object_ids_M = obtain_rollout_trajs_in_world(batch, result_M, traj_noise_std)
      
      scenario_rollouts = obtain_waymo_scenario_rollouts(waymo_scene, rollout_trajs_in_world_M, object_ids_M, ego_sim_agent_id)

      print(f'process({os.getpid()}): rollout scene {scene_idx} finished: ', time.time() - start, flush=True)
      print(f'process({os.getpid()}) memory usage: ', check_mem_usage(os.getpid()), flush=True)

      if save_rollout:
        save_string = scenario_rollouts.SerializeToString()
        with open(rollout_file, 'wb') as f:
          f.write(save_string)
        print(f'process({os.getpid()}): save rollout scene {scene_idx} to {rollout_file}', flush=True)

      if save_metric:
        print(f'process({os.getpid()}): start compute metric scene {scene_idx}', flush=True)
        start = time.time()
        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(waymo_config, waymo_scene, scenario_rollouts)
        print(f'process({os.getpid()}): compute metric scene {scene_idx} finished', time.time() - start, flush=True)

        metric_dict = {}
        for line in str(scenario_metrics).split('\n'):
          if ':' not in line:
            continue
          key, value = line.split(':')
          key = key.strip()
          value = value.strip()
          metric_dict[key] = value

        with open(metric_file, 'w') as f:
          json.dump(metric_dict, f)

        print(f'process({os.getpid()}): save metric scene {scene_idx} to {metric_file}', flush=True)

    except Exception as e:
      print(f'process({os.getpid()}): error in scene {scene_idx}: {e}', flush=True)