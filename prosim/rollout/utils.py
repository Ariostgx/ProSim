import sys
import tqdm
import pickle
import torch
import numpy as np
from typing import List, Optional, Tuple, Union


from prosim.models.utils.visualization import vis_rollout_traj_pred

from trajdata.data_structures.state import StateArray
from prosim.models.utils.data import get_agent_pos_dict
from prosim.models.utils.geometry import wrap_angle
from prosim.models.utils.data import extract_agent_obs_from_center_obs
from prosim.dataset.data_utils import rotate

from trajdata.utils.arr_utils import transform_coords_np, transform_angles_np, transform_matrices

def get_agent_gt_next_state(idx, obs):
    curr_yaw = obs.curr_agent_state[idx].heading.item()
    curr_pos = obs.curr_agent_state[idx].position.numpy()
    world_from_agent = np.array(
        [
            [np.cos(curr_yaw), np.sin(curr_yaw)],
            [-np.sin(curr_yaw), np.cos(curr_yaw)],
        ]
    )
    next_state = np.zeros((4,))
    if obs.agent_fut_len[idx] < 1:
        next_state[:2] = curr_pos
        yaw_ac = 0
    else:
        next_state[:2] = (
            obs.agent_fut[idx, 0].position.numpy() @ world_from_agent
            + curr_pos
        )
        yaw_ac = obs.agent_fut[idx, 0].heading.item()

    next_state[-1] = curr_yaw + yaw_ac

    return StateArray.from_array(next_state, "x,y,z,h")

def get_agent_stop_state(idx, obs):
    curr_yaw = obs.curr_agent_state[idx].heading.item()
    curr_pos = obs.curr_agent_state[idx].position.numpy()

    next_state = np.zeros((4,))
    next_state[:2] = curr_pos
    next_state[-1] = curr_yaw
    return StateArray.from_array(next_state, "x,y,z,h")

def transform_traj_to_global(action_dicts, control_agents, center_obs):
    traj_type_to_trans = ['motion_pred', 'policy_goal', 'pred_goal']

    agent_obs = extract_agent_obs_from_center_obs(control_agents, center_obs)

    traslation = agent_obs['hist'][0, :, -1].as_format('x,y')
    rotation = agent_obs['hist'][0, :, -1].as_format('h')
    center_from_control_tf = transform_matrices(rotation[:, 0], traslation)

    world_from_center_tf = torch.linalg.inv(center_obs.agents_from_world_tf)[0]

    step_global_trajs = {}
    for idx, agent_name in enumerate(control_agents):

        if agent_name not in action_dicts:
            continue

        world_from_control_tf = world_from_center_tf @ center_from_control_tf[idx]
        world_from_control_tf = world_from_control_tf.detach().cpu().numpy()

        step_global_trajs[agent_name] = {}
        for traj_type in action_dicts[agent_name].keys():
            if traj_type not in traj_type_to_trans:
                continue

            local_traj = action_dicts[agent_name][traj_type].detach().cpu().double().numpy()
            local_xy = local_traj[..., :2]
            global_xy = transform_coords_np(local_xy, world_from_control_tf)
            global_traj = [global_xy]

            if traj_type == 'motion_pred':
                local_angle = local_traj[..., -1]
                global_angle = transform_angles_np(local_angle, world_from_control_tf)
                global_traj.append(global_angle[..., None])
            
            global_traj = np.concatenate(global_traj, axis=-1)
            step_global_trajs[agent_name][traj_type] = global_traj

    return step_global_trajs

def get_agent_planed_next_state(curr_yaw, curr_pos, next_acc):
    world_from_agent = np.array(
        [
            [np.cos(curr_yaw), np.sin(curr_yaw)],
            [-np.sin(curr_yaw), np.cos(curr_yaw)],
        ]
    )
    next_state = np.zeros((4,))
    pos_acc = next_acc[:2]

    if next_acc.shape[0] == 3:
        yaw_ac = next_acc[-1]
    else:
        yaw_ac = 0

    next_state[:2] = (
        pos_acc @ world_from_agent
        + curr_pos
    )

    next_state[-1] = wrap_angle(curr_yaw + yaw_ac)

    return StateArray.from_array(next_state, "x,y,z,h")


def get_policy_emb(model, agent_names, ego_obs, prompt_result):
  prompt_dict = {'motion_pred': prompt_result}
  policy_emb, scene_emb = model.get_policy(prompt_dict, batch=ego_obs)

  agent_policy_dict = {}
  for idx, agent_name in enumerate(agent_names):
    policy_dict = {}
    for key in policy_emb['motion_pred']:
        policy_dict[key] = policy_emb['motion_pred'][key][0, idx]
    
    if 'policy_goal_rel' in prompt_result:
        policy_dict['gt_goal'] = prompt_result['policy_goal_rel'][idx]

    agent_policy_dict[agent_name] = policy_dict

  return agent_policy_dict, scene_emb

def create_affine_transform(rotation_angle, translation_vector):
    cos_theta = torch.cos(rotation_angle)
    sin_theta = torch.sin(rotation_angle)
    tx, ty = translation_vector

    # Creating the affine transformation matrix directly
    transform_matrix = torch.tensor([
        [cos_theta, -sin_theta, tx],
        [sin_theta,  cos_theta, ty],
        [0,         0,         1]
    ])

    return transform_matrix

def get_curr_obs_to_policy_obs_tf(curr_obs, policy_obs):
    c_pos = curr_obs.curr_agent_state.position[0].double()
    c_yaw = curr_obs.curr_agent_state.heading[0].double()
    p_pos = policy_obs.curr_agent_state.position[0].double()
    p_yaw = policy_obs.curr_agent_state.heading[0].double()

    curr_to_world = create_affine_transform(c_yaw, c_pos)
    policy_to_world = create_affine_transform(p_yaw, p_pos)
    world_to_policy = torch.inverse(policy_to_world.double())

    policy_from_curr = torch.mm(world_to_policy.double(), curr_to_world.double())

    return policy_from_curr.cpu().numpy()

def convert_curr_obs_to_policy_obs(curr_obs, policy_from_curr_tf):
    # convert observation from current frame to policy frame
    new_obs = curr_obs.extras['init_obs']

    device = new_obs['position'].device

    new_obs['position'] = transform_coords_np(new_obs['position'].double().cpu().numpy(), policy_from_curr_tf)
    new_obs['heading'] = transform_angles_np(new_obs['heading'].double().cpu().numpy(), policy_from_curr_tf)

    new_obs['position'] = torch.tensor(new_obs['position'], device=device).float()
    new_obs['heading'] = torch.tensor(new_obs['heading'], device=device).float()

    return new_obs

def convert_agent_pos_dict_to_policy_obs(agent_names, center_obs, policy_from_curr_tf):
    agent_obs = extract_agent_obs_from_center_obs(agent_names, center_obs)
    batch_pos_data = get_agent_pos_dict(agent_obs['hist'])

    device = batch_pos_data['position'].device

    batch_pos_data['position'] = transform_coords_np(batch_pos_data['position'].double().cpu().numpy(), policy_from_curr_tf)[0]
    batch_pos_data['heading'] = transform_angles_np(batch_pos_data['heading'].double().cpu().numpy(), policy_from_curr_tf)[0]

    batch_pos_data['position'] = torch.tensor(batch_pos_data['position'], device=device).float()
    batch_pos_data['heading'] = torch.tensor(batch_pos_data['heading'], device=device).float()

    return batch_pos_data

from prosim.dataset.data_utils import transform_to_frame_offset_rot

def get_action_batch(model, policy_emb_dict, scene_emb, agent_names, center_obs, policy_obs, latent_state):

  # transform all the position embeding to the frame of the policy_obs
  # update scene_emb with the current observation
  policy_from_curr_tf = get_curr_obs_to_policy_obs_tf(center_obs, policy_obs)
  curr_obs = convert_curr_obs_to_policy_obs(center_obs, policy_from_curr_tf)

  scene_emb = model.scene_encoder.update_scene_emb(scene_emb, curr_obs)
  
  agent_obs_data = [model._get_io_token('', scene_emb, 0, 0, 'obs') for _ in range(len(agent_names))]
  batch_obs_data = model._collate_batch_dict(agent_obs_data)

  agent_map_data = [model._get_io_token('', scene_emb, 0, 0, 'map') for _ in range(len(agent_names))]
  batch_map_data = model._collate_batch_dict(agent_map_data)

  agent_policy_emb = [policy_emb_dict[agent_name] for agent_name in agent_names]
  batch_policy_emb = model._collate_batch_dict(agent_policy_emb)

  batch_pos_data = convert_agent_pos_dict_to_policy_obs(agent_names, center_obs, policy_from_curr_tf)

  pair_names = [f'0-{agent}-0' for agent in agent_names]

  model_output = model.get_action(batch_policy_emb, batch_obs_data, batch_map_data, batch_pos_data, [pair_names], latent_state)

  if 'latent_state' in model_output:
    latent_state = model_output['latent_state']
  else:
    latent_state = None

  best_acc_dict, action_dicts = {}, {}

  for idx, agent_name in enumerate(agent_names):
    action_dist = {key: model_output[key][idx].detach().cpu() for key in model_output if key != 'latent_state'}

    k_idx = torch.argmax(action_dist['motion_prob'])
    best_motion = action_dist['motion_pred'][k_idx]
    best_acc_dict[agent_name] = torch.diff(best_motion, dim=0)
    best_acc_dict[agent_name] = torch.cat((best_motion[:1], best_acc_dict[agent_name])).numpy()

    action_dicts[agent_name] = action_dist

  return best_acc_dict, action_dicts, scene_emb, latent_state

def sample_pred_goal(emd_dict, config, policy_obs, prompt_dict, goal_dict):
  # convert predicted goal point for all the agent to policy_obs's frame

  agent_names = emd_dict.keys()
  
  init_position = prompt_dict['position'][0]
  init_heading = prompt_dict['heading'][0]

  for idx, agent_name in enumerate(agent_names):

    if config.MODEL.DECODER.GOAL_PRED.HEATMAP_PRED:
        goal_map_idx = emd_dict[agent_name]['goal'].detach().argmax()
        map_positions = policy_obs.extras['init_map']['position'][0][..., :2].squeeze(1)
        pred_goal = map_positions[goal_map_idx]

        pred_goal_rel = pred_goal - init_position[idx]
        pred_goal_rel = rotate(pred_goal_rel[...,0], pred_goal_rel[...,1], -init_heading[idx])
    
    else:
        pred_goal_rel = emd_dict[agent_name]['goal'].detach()
        pred_goal = rotate(pred_goal_rel[...,0], pred_goal_rel[...,1], init_heading[idx]) + init_position[idx]
    
    goal_dict[agent_name]['pred_goal_rel'] = pred_goal_rel.squeeze(0)
    goal_dict[agent_name]['pred_goal'] = pred_goal.squeeze(0)

  return goal_dict

def set_goal_context(agent_policy_embs, goal_dict, config):
    context_cfg = config.MODEL.POLICY.ACT_DECODER.CONTEXT
    
    # use pred goal for rollout
    if context_cfg.GT_GOAL:
        for agent_name in agent_policy_embs:
            agent_policy_embs[agent_name]['gt_goal'] = goal_dict[agent_name]['pred_goal_rel']
    
    return agent_policy_embs

def angle_wrap(
    radians: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """This function wraps angles to lie within [-pi, pi).

    Args:
        radians (np.ndarray): The input array of angles (in radians).

    Returns:
        np.ndarray: Wrapped angles that lie within [-pi, pi).
    """
    return (radians + np.pi) % (2 * np.pi) - np.pi

def batch_derivative_of(states, dt = 1.):
    """
    states: [..., T, state_dim]
    dt: time difference between states in input trajectory
    """
    # diff = states[..., 1:, :] - states[..., :-1, :]
    # Add first state derivative
    if isinstance(states, torch.Tensor):
        diff = torch.diff(states, 1, dim=-2)        
        diff = torch.cat((diff[..., :1, :], diff), dim=-2)
    else:
        diff = np.diff(states, 1, axis=-2)
        diff = np.concatenate((diff[..., :1, :], diff), axis=-2)
    return diff / dt

def traj_xyhv_to_xyhvv(traj):
    if isinstance(traj, torch.Tensor):
        x, y, h, v = torch.split(traj, 1, dim=-1)
        vx = v * torch.cos(h)
        vy = v * torch.sin(h)
        return torch.cat((x, y, h, vx, vy), dim=-1)
    else:
        x, y, h, v = np.split(traj, 4, axis=-1)
        vx = v * np.cos(h)
        vy = v * np.sin(h)
        return np.concatenate((x, y, h, vx, vy), axis=-1)

def traj_xyhvv_to_pred(traj, dt):
    # Input: [x, y, h, vx, vy] 
    # Output prediction state: ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sintheta', 'costheta']
    if isinstance(traj, torch.Tensor):
        x, y, h, vx, vy = torch.split(traj, 1, dim=-1)
        ax = batch_derivative_of(vx, dt=dt)
        ay = batch_derivative_of(vy, dt=dt)
        pred_state = torch.cat((
            x, y, vx, vy, ax, ay, torch.sin(h), torch.cos(h)
        ), dim=-1)
    else:
        x, y, h, vx, vy = np.split(traj, 5, axis=-1)
        ax = batch_derivative_of(vx, dt=dt)
        ay = batch_derivative_of(vy, dt=dt)
        pred_state = np.concatenate((
            x, y, vx, vy, ax, ay, np.sin(h), np.cos(h)
        ), axis=-1)        
    return pred_state

def batch_nd_transform_points_np(points: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim - 2)) + [Mat.ndim - 1] + [Mat.ndim - 2]
    Mat = np.transpose(Mat, batch)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")

def batch_nd_transform_points_pt(
    points: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    ndim = Mat.shape[-1] - 1
    Mat = torch.transpose(Mat, -1, -2)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    elif points.ndim == Mat.ndim + 1:
        return (
            (
                points[..., np.newaxis, :]
                @ Mat[..., np.newaxis, np.newaxis, :ndim, :ndim]
            )
            + Mat[..., np.newaxis, np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_angles_np(angles: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = np.arctan2(sin_vals, cos_vals)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_angles_pt(
    angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = torch.arctan2(sin_vals, cos_vals)
    if rot_angle.ndim > angles.ndim:
        raise ValueError("wrong shape")
    while rot_angle.ndim < angles.ndim:
        rot_angle = rot_angle.unsqueeze(-1)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_points_angles_np(
    points_angles: np.ndarray, Mat: np.ndarray
) -> np.ndarray:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_np(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_np(points_angles[..., 2:3], Mat)
    points_angles = np.concatenate([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_points_angles_pt(
    points_angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_pt(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_pt(points_angles[..., 2:3], Mat)
    points_angles = torch.concat([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_xyvvaahh_pt(traj_xyvvaahh: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """
    traj_xyvvaahh: [..., state_dim] where state_dim = [x, y, vx, vy, ax, ay, sinh, cosh]
    This is the state representation used in AgentBatch and SceneBatch.
    """
    rot_only_tf = tf.clone()
    rot_only_tf[..., :2, -1] = 0.

    xy, vv, aa, hh = torch.split(traj_xyvvaahh, (2, 2, 2, 2), dim=-1)
    xy = batch_nd_transform_points_pt(xy, tf)
    vv = batch_nd_transform_points_pt(vv, rot_only_tf)
    aa = batch_nd_transform_points_pt(aa, rot_only_tf)
    # hh: sinh, cosh instead of cosh, sinh, so we use flip
    hh = batch_nd_transform_points_pt(hh.flip(-1), rot_only_tf).flip(-1)

    return torch.concat((xy, vv, aa, hh), dim=-1)

def vis_rollout_frame(obs, action_dicts, action_idx, step_global_trajs):
    obs.neigh_hist[:, :, :, 2] = 0
    obs.neigh_fut[:, :, :, 2] = 0
    action_vis = vis_rollout_traj_pred(obs, action_dicts, action_idx, step_global_trajs)
    frame = np.concatenate(action_vis, axis=1)
    return frame

def get_curr_state_in_world(control_agents, center_obs):
    center_pos = center_obs.curr_agent_state.position[0]
    center_yaw = center_obs.curr_agent_state.heading[0]

    curr_to_world = create_affine_transform(center_yaw, center_pos)

    agent_obs = extract_agent_obs_from_center_obs(control_agents, center_obs)
    agent_state = agent_obs['hist'][0, :, -1]

    agent_xy = agent_state.as_format('x,y')
    agent_xy = transform_coords_np(agent_xy.cpu().numpy(), curr_to_world.cpu().numpy())

    agent_h = agent_state.as_format('h')
    agent_h = transform_angles_np(agent_h.cpu().numpy(), curr_to_world.cpu().numpy())

    curr_state = np.zeros((len(control_agents), 4))
    curr_state[:, :2] = agent_xy
    curr_state[:, -1:] = agent_h

    result = {}
    for idx, a_id in enumerate(control_agents):
        result[a_id] = StateArray.from_array(curr_state[idx], "x,y,z,h")

    return result

def rollout_scene_loop(sim_scene, pl_module, control_agents, start_frame, max_step, center_agent, config, enable_vis, prompt_value, verbose=False):
    # TODO: delete this function
    return None


# def rollout_scene_loop(sim_scene, pl_module, control_agents, start_frame, max_step, center_agent, config, enable_vis, prompt_value, verbose=False):
#     import time
    
#     control_agent_info = [agent for agent in sim_scene.agents if agent.name in control_agents]
    
#     prompt_cls = config.ROLLOUT.PROMPT
#     prompt_generator = prompt_generators[prompt_cls](config.PROMPT[prompt_cls.upper()])

#     vis_frames = []

#     rollout_global_states = []
    
#     curr_state_in_world = None

#     for t in tqdm.trange(start_frame, max_step):
#         to_update_policy = (t - start_frame) % config.ROLLOUT.POLICY.POLICY_FREQ == 0
#         to_run_policy = (t - start_frame) % config.ROLLOUT.POLICY.REPLAN_FREQ == 0
        
#         if to_update_policy:
#             start_prompt = time.time()
#             policy_obs = sim_scene.get_obs(agent_names=[center_agent], get_vector_map=True)
#             end_prompt = time.time()
#             policy_obs.to(pl_module.device)
#             if verbose:
#                 print(f'get prompt obs time: {end_prompt - start_prompt}')
            
#             start_prompt_model = time.time()

#             prompt_dict = prompt_generator.prompt_for_rollout_batch(control_agents, policy_obs, prompt_value)
#             agent_policy_embs, scene_emb = get_policy_emb(pl_module, control_agents, policy_obs, prompt_dict)
            
#             # set goal prompt or goal prediction result
#             goal_dict = {a_id: {} for a_id in control_agents}

#             if 'goal' in prompt_cls:
#                 for idx, a_id in enumerate(control_agents):
#                     goal_dict[a_id]['policy_goal'] = prompt_dict['policy_goal'][idx]

#             if config.MODEL.DECODER.GOAL_PRED.ENABLE:
#                 goal_dict = sample_pred_goal(agent_policy_embs, config, policy_obs, prompt_dict, goal_dict)
#                 agent_policy_embs = set_goal_context(agent_policy_embs, goal_dict, config)
            

#             latent_state = None

#             end_prompt_model = time.time()

#             if verbose:
#                 print(f'prompt model time: {end_prompt_model - start_prompt_model}')
#                 print(f'----- new policy embedding at step {t} -----')

        
#         if  to_run_policy:
#             if to_update_policy:
#                 center_obs = policy_obs
#             else:
#                 start_get_obs = time.time()
#                 center_obs = sim_scene.get_obs(agent_names=[center_agent], get_vector_map=False)
#                 center_obs.to(pl_module.device)
#                 end_get_obs = time.time()

#                 if verbose:
#                     print(f'get obs time: {end_get_obs - start_get_obs}')

#             start_policy = time.time()

#             pred_acc_dict, action_dicts, scene_emb, latent_state = get_action_batch(pl_module, agent_policy_embs, scene_emb, control_agents, center_obs, policy_obs, latent_state)
        
#             action_idx = 0

#             end_policy = time.time()

#             if verbose:
#                 print(f'policy step time: {end_policy - start_policy}')

#         if curr_state_in_world is None:
#             curr_state_in_world = get_curr_state_in_world(control_agents, center_obs)
#         else:
#             curr_state_in_world = next_state_in_world


#         next_state_in_world = {}
#         for idx, a_id in enumerate(control_agents):
#             next_acc = pred_acc_dict[a_id][action_idx]
#             curr_yaw = curr_state_in_world[a_id].heading[0]
#             curr_pos = curr_state_in_world[a_id].position

#             next_state_in_world[a_id] = get_agent_planed_next_state(curr_yaw, curr_pos, next_acc)

#         action_idx += 1

#         start_step = time.time()
#         sim_scene.step(next_state_in_world, return_obs=False, use_nr_background=True, control_agent_info=control_agent_info)        
#         end_step = time.time()

#         if verbose:
#             print(f'step time: {end_step - start_step}')

#         rollout_global_states.append(next_state_in_world)

#         if enable_vis and (t - start_frame) % config.ROLLOUT.VISUALIZE.GIF_FREQ == 0:
            
#             if to_update_policy:
#                 global_goal = transform_traj_to_global(goal_dict, control_agents, policy_obs)
            
#             step_global_trajs = transform_traj_to_global(action_dicts, control_agents, center_obs)
#             for a_id in global_goal:
#                 step_global_trajs[a_id].update(global_goal[a_id])

#             vis_obs = sim_scene.get_obs(agent_names=[center_agent], get_vector_map=False, get_raster_map=True)
#             frame = vis_rollout_frame(vis_obs, action_dicts, action_idx, step_global_trajs)
#             vis_frames.append(frame)
    
#     return sim_scene, vis_frames, rollout_global_states