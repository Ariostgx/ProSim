import torch
# import tensorflow as tf
import numpy as np

from prosim.dataset.format_utils import BatchDataDict, InputMaskData
from prosim.rollout.waymo_utils import get_waymo_file_template, get_waymo_scene_object, joint_scene_from_states, plot_waymo_gt_trajectory, rollout_states_to_joint_scene, plot_waymo_rollout_trajectory
from prosim.models.utils.geometry import wrap_angle, batch_rotate_2D
from prosim.rollout.utils import batch_nd_transform_points_pt, batch_nd_transform_angles_pt
# from waymo_open_dataset.utils.sim_agents import submission_specs
# from waymo_open_dataset.protos import sim_agents_submission_pb2

def get_waymo_specification(batch, config):
  scene_name = batch.scene_ids[0]

  scene_template = get_waymo_file_template(config)
  waymo_scene = get_waymo_scene_object(scene_name, scene_template)
  sim_agent_ids = submission_specs.get_sim_agent_ids(waymo_scene)

  batch_prompt_info = batch.extras['prompt']['motion_pred']
  batch_agent_ids = batch_prompt_info['agent_ids'][0]

  valid_idx = []
  for i, agent_id in enumerate(batch_agent_ids):
    if 'ego' in agent_id or int(agent_id) in sim_agent_ids:
      valid_idx.append(i)
  
  print('batch_agent_ids: ', batch_agent_ids)
  print('valid_idx: ', valid_idx)
  print('sim_agent_ids: ', sim_agent_ids)
  print(len(valid_idx), len(sim_agent_ids))

  assert len(valid_idx) == len(sim_agent_ids)

  batch_agent_ids = [batch_agent_ids[i] for i in valid_idx]
  missed_agent_ids = [sim_agent_id for sim_agent_id in sim_agent_ids if str(sim_agent_id) not in batch_agent_ids]
  
  assert len(missed_agent_ids) == 1

  ego_sim_agent_id = missed_agent_ids[0]

  # subsample the batch_prompt info to only include the sim_agent_ids
  for key in batch_prompt_info:
    if type(batch_prompt_info[key]) == list:
      batch_prompt_info[key][0] = [batch_prompt_info[key][0][i] for i in valid_idx]
    elif type(batch_prompt_info[key]) == torch.Tensor:
      batch_prompt_info[key] = batch_prompt_info[key][:, valid_idx]
  
  # if len(missed_agent_ids) > 0:
  #   print('missed agent ids: ', missed_agent_ids)

  #   invalid_tracks = [track for track in waymo_scene.tracks if track.id in missed_agent_ids]
  #   for track in invalid_tracks:
  #     print(track.id)
  #     valid_state = [state for state in track.states if state.valid]
  #     print('num valid states: ', len(valid_state))
  
  return batch, waymo_scene, ego_sim_agent_id

def replica_batch_for_parallel_rollout(scene_embs, policy_emds, prompt_encs, policy_agent_ids, agent_trajs, batch, M):
  # replicate scene embs
  scene_embs_M = {}
  for name in ['obs_mask', 'map_mask', 'scene_pos', 'scene_ori', 'scene_tokens']:
    scene_embs_M[name] = scene_embs[name].repeat(M, 1)
  scene_embs_M['max_map_num'] = scene_embs['max_map_num']
  scene_embs_M['max_agent_num'] = scene_embs['max_agent_num']
  scene_embs_M['scene_type'] = scene_embs['scene_type'].repeat(M)
  N = scene_embs['scene_batch_idx'].shape[0]
  device = scene_embs['scene_batch_idx'].device
  scene_embs_M['scene_batch_idx'] = torch.arange(M, device=device)[:, None].repeat(1, N).reshape(-1)

  if policy_emds is None:
    policy_emds_M = None
  else:
    policy_emds_M = {'motion_pred': {}}

    for name in policy_emds['motion_pred'].keys():
      dim_num = policy_emds['motion_pred'][name].ndim
      reshape_dims = [M] + [1] * (dim_num - 1)
      policy_emds_M['motion_pred'][name] = policy_emds['motion_pred'][name].repeat(*reshape_dims)
  

  # replicate policy_agent_ids
  policy_agent_ids_M = {}
  policy_agent_ids_M['motion_pred'] = [policy_agent_ids['motion_pred'][0]] * M

  # replicate agent_trajs
  agent_trajs_M = {'motion_pred': {}}
  agent_trajs_M['motion_pred']['traj'] = agent_trajs['motion_pred']['traj'].repeat(M, 1, 1, 1)
  agent_trajs_M['motion_pred']['init_pos'] = agent_trajs['motion_pred']['init_pos'].repeat(M, 1, 1)
  agent_trajs_M['motion_pred']['init_heading'] = agent_trajs['motion_pred']['init_heading'].repeat(M, 1, 1)
  agent_trajs_M['motion_pred']['last_step'] = agent_trajs['motion_pred']['last_step']

  if 'vel' in agent_trajs['motion_pred']:
    agent_trajs_M['motion_pred']['vel'] = agent_trajs['motion_pred']['vel'].repeat(M, 1, 1, 1)

  # replicate prompt_encs
  prompt_encs_M = {'motion_pred': {}}
  prompt_encs_M['motion_pred']['prompt'] = prompt_encs['motion_pred']['prompt'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['prompt_mask'] = prompt_encs['motion_pred']['prompt_mask'].repeat(M, 1)
  prompt_encs_M['motion_pred']['position'] = prompt_encs['motion_pred']['position'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['heading'] = prompt_encs['motion_pred']['heading'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['agent_type'] = prompt_encs['motion_pred']['agent_type'].repeat(M, 1)
  prompt_encs_M['motion_pred']['prompt_emd'] = prompt_encs['motion_pred']['prompt_emd'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['agent_ids'] = prompt_encs['motion_pred']['agent_ids'][0] * M

  # replicate batch['extras']['fut_obs']
  batch_fut_obs_M = {}
  for step in batch.extras['fut_obs'].keys():
    fut_obs = batch.extras['fut_obs'][step]
    fut_obs_M = {}
    for key in fut_obs.keys():
      if type(fut_obs[key]) == list:
        fut_obs_M[key] = fut_obs[key] * M
      elif type(fut_obs[key]) == torch.Tensor:
        ndim = fut_obs[key].ndim
        repeat_dims = [M] + [1] * (ndim -1)
        fut_obs_M[key] = fut_obs[key].repeat(*repeat_dims)
    
    batch_fut_obs_M[step] = InputMaskData.from_dict(fut_obs_M)

  batch.extras['fut_obs'] = BatchDataDict(batch_fut_obs_M)

  return scene_embs_M, policy_emds_M, prompt_encs_M, policy_agent_ids_M, agent_trajs_M, batch

def sample_M_goal_cond_to_batch(batch, sample_result, top_K, M, stop_smooth_num=5.0):
  # assume sample_result is with 1 scene (bz=1)
  device = batch.extras['prompt']['motion_pred']['prompt'].device
  
  goal_inputs_M = []
  prompt_idxs_M = []

  print('smoothing stopping action with distance: ', stop_smooth_num)
  
  for b in range(M):
      goal_inputs_b = []
      prompt_idxs_b = []
      
      for pidx, aname in enumerate(batch.extras['prompt']['motion_pred']['agent_ids'][0]):
          pair_name = f'0-{aname}-0'
          if pair_name in sample_result['motion_pred']['pair_names']:
              pred_idx = sample_result['motion_pred']['pair_names'].index(pair_name)
              pred_goal_K = sample_result['motion_pred']['goal_point'][pred_idx]
              pred_goal_K_prob = sample_result['motion_pred']['goal_prob'][pred_idx]
      
              top_k_idx = torch.argsort(-pred_goal_K_prob)[:top_K]
              
              select_idx = top_k_idx[torch.randperm(top_K)[0]]
              select_goal = pred_goal_K[select_idx]

              if torch.abs(select_goal[0]) < stop_smooth_num and torch.abs(select_goal[1]) < stop_smooth_num:
                select_goal[0] = 0.0
                select_goal[1] = 0.0
      
              goal_inputs_b.append(torch.tensor([select_goal[0], select_goal[1], 80.0]))
              prompt_idxs_b.append(torch.tensor([pidx]))
      
      goal_inputs_b = torch.stack(goal_inputs_b)
      prompt_idxs_b = torch.stack(prompt_idxs_b)
  
      goal_inputs_M.append(goal_inputs_b)
      prompt_idxs_M.append(prompt_idxs_b)
  
  goal_inputs_M = torch.stack(goal_inputs_M).to(device)
  prompt_idxs_M = torch.stack(prompt_idxs_M).to(device)
  
  N = goal_inputs_M.shape[1]
  
  mask_M = torch.ones(M, N, dtype=torch.bool).to(device)
  prompt_mask_M = torch.ones(M, N, dtype=torch.bool).to(device)
  
  caption_str = 'show as green cross'
  
  goal_cond_M = {'input': goal_inputs_M, 'mask': mask_M, 'prompt_idx': prompt_idxs_M, 'prompt_mask': prompt_mask_M, 'caption_str': caption_str}
  
  batch.extras['condition'].all_cond = {'goal': goal_cond_M}

  return batch

def parallel_rollout_batch(batch, M, model, top_K=3, sampler_model=None, smooth_dist=5.0):
  with torch.no_grad():
    import time

    # start = time.time()
    scene_embs = model.encode_scene(batch)
    # print('encode scene: ', time.time() - start)

    # start = time.time()
    prompt_encs = model.encode_prompt(batch)
    # print('encode prompt', time.time() - start)

    # start = time.time()
    # print('decode policy: ', time.time() - start)

    policy_agent_ids = {task: batch.extras['prompt'][task]['agent_ids'] for task in ['motion_pred']}
    all_t_indices = sorted(batch.extras['all_t_indices'].cpu().numpy().tolist())
    agent_trajs = model.init_agent_trajs(policy_agent_ids, batch, all_t_indices)


    if sampler_model is None:
      policy_emds = model.decode_policy(batch, scene_embs, prompt_encs)

      scene_embs_M, policy_emds_M, _, policy_agent_ids_M, agent_trajs_M, batch = replica_batch_for_parallel_rollout(scene_embs, policy_emds, prompt_encs, policy_agent_ids, agent_trajs, batch, M)
    else:
      print(f'using sampler model to get top_K={top_K} goal conditions for {M} replicas with smooth_dist={smooth_dist}')

      sample_result = sampler_model.forward(batch, 'val')

      # print(sample_result['motion_pred']['pair_names'])
      # print(sample_result['motion_pred']['goal_point'])
      # print(sample_result['motion_pred']['goal_prob'])

      scene_embs_M, _, prompt_encs_M, policy_agent_ids_M, agent_trajs_M, batch = replica_batch_for_parallel_rollout(scene_embs, None, prompt_encs, policy_agent_ids, agent_trajs, batch, M)

      batch = sample_M_goal_cond_to_batch(batch, sample_result, top_K, M, stop_smooth_num=smooth_dist)

      policy_emds_M = model.decode_policy(batch, scene_embs_M, prompt_encs_M)

    # end = time.time()
    # print('replica_batch_for_parallel_rollout: ', end - start)

    # print(all_t_indices)

    # start = time.time()
    model.mode = 'rollout'
    result_M = model.rollout_batch(batch, scene_embs_M, policy_emds_M, policy_agent_ids_M, agent_trajs_M, all_t_indices, 'rollout')
    # print('rollout_batch full: ', time.time() - start)
  
  return result_M

def obtain_rollout_trajs_in_world(batch, result_M, noise_std=0.0):
  batch_ids = []
  object_ids = []

  trajs = []
  init_pos = []
  init_heads = []

  for agent_name, results in result_M['motion_pred']['rollout_trajs'].items():
    batch_id = int(agent_name.split('-')[0])
    object_id = agent_name.split('-')[1]
    
    batch_ids.append(batch_id)
    object_ids.append(object_id)
    trajs.append(results['traj'])
    init_pos.append(results['init_pos'])
    init_heads.append(results['init_heading'])

  batch_ids = torch.tensor(batch_ids)
  trajs = torch.stack(trajs, axis=0)
  init_pos = torch.stack(init_pos, axis=0)
  init_heads = torch.stack(init_heads, axis=0)

  print('trajs shape: ', trajs.shape)

  if noise_std > 0.0:
    print('WARNING: adding noise to the rollout trajectories with std: ', noise_std)
    trajs[:, :, :2] += torch.randn_like(trajs[:, :, :2]) * noise_std

  # transform all trajectories to the starting frame's coordinate system
  xys_in_center = batch_rotate_2D(trajs[:, :, :2], init_heads) + init_pos[:, None]
  hs = torch.arctan2(trajs[:, :, 2], trajs[:, :, 3])
  hs_in_centers = wrap_angle(hs + init_heads)

  # transform all trajectories to the world coordinate system
  center_to_world_tf = batch.centered_world_from_agent_tf[0]

  xys_in_world = batch_nd_transform_points_pt(xys_in_center, center_to_world_tf)
  hs_in_world = batch_nd_transform_angles_pt(hs_in_centers, center_to_world_tf)

  rollout_trajs_in_world = torch.cat([xys_in_world, hs_in_world[:, :, None]], axis=-1)

  rollout_trajs_in_world_M = []
  object_ids_M = []

  for batch_id in set(batch_ids.tolist()):
    batch_mask = batch_ids == batch_id
    
    rollout_trajs_in_world_M.append(rollout_trajs_in_world[batch_mask].detach().cpu().numpy())
    object_ids_M.append([object_ids[i] for i in range(len(object_ids)) if batch_mask[i]])
  
  return rollout_trajs_in_world_M, object_ids_M

def joint_scene_from_rollout(waymo_scene, rollout_trajs, object_ids, ego_sim_agent_id):
  joint_trajs = rollout_trajs

  ego_idx = object_ids.index('ego')
  control_ids = object_ids.copy()
  control_ids[ego_idx] = str(ego_sim_agent_id)
  control_ids = [int(agent) for agent in control_ids]

  # # use the z from the first frame of the waymo scene
  scene_track_ids = [track.id for track in waymo_scene.tracks]
  track_indices = [scene_track_ids.index(int(agent)) for agent in control_ids]
  z_start = np.array([waymo_scene.tracks[idx].states[submission_specs.CURRENT_TIME_INDEX].center_z for idx in track_indices])
  z_trajs = np.ones_like(joint_trajs[..., :1]) * z_start[:, None, None]

  joint_trajs = np.concatenate([joint_trajs[..., :2], z_trajs, joint_trajs[..., 2:]], axis=-1)

  joint_scene = joint_scene_from_states(joint_trajs, control_ids)

  return joint_scene

def obtain_waymo_scenario_rollouts(waymo_scene, rollout_trajs_in_world_M, object_ids_M, ego_sim_agent_id):
  M = len(rollout_trajs_in_world_M)

  joint_scenes = []
  for j in range(32 // M):
    for i in range(M):
      joint_scene = joint_scene_from_rollout(waymo_scene, rollout_trajs_in_world_M[i], object_ids_M[i], ego_sim_agent_id)
      joint_scenes.append(joint_scene)

  batch_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
    joint_scenes=joint_scenes, scenario_id=waymo_scene.scenario_id)

  submission_specs.validate_scenario_rollouts(batch_rollouts, waymo_scene)

  return batch_rollouts
