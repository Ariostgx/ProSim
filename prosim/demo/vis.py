from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Polygon
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.state import StateTensor
from trajdata.maps import RasterizedMap


def draw_agent(
    ax: Axes,
    agent_type: AgentType,
    agent_state: StateTensor,
    agent_extent: Tensor,
    agent_to_world_tf: Tensor,
    **kwargs,
) -> None:
    """Draws a path with the correct location, heading, and dimensions onto the given axes

    Args:
        ax (Axes): _description_
        agent_type (AgentType): _description_
        agent_state (Tensor): _description_
        agent_extent (Tensor): _description_
        agent_to_world_tf (Tensor): _description_
    """

    if torch.any(torch.isnan(agent_extent)):
        if agent_type == AgentType.VEHICLE:
            length = 4.3
            width = 1.8
        elif agent_type == AgentType.PEDESTRIAN:
            length = 0.5
            width = 0.5
        elif agent_type == AgentType.BICYCLE:
            length = 1.9
            width = 0.5
        else:
            length = 1.0
            width = 1.0
    else:
        length = agent_extent[0].item()
        width = agent_extent[1].item()

    xy = agent_state.position
    heading = agent_state.heading

    patch = FancyBboxPatch([-length / 2, -width / 2], length, width, **kwargs)
    transform = (
        mtransforms.Affine2D().rotate(heading[0].item()).translate(xy[0], xy[1])
        + mtransforms.Affine2D(matrix=agent_to_world_tf.cpu().numpy())
        + ax.transData
    )
    patch.set_transform(transform)

    kwargs["label"] = None
    size = 1.0
    angles = [0, 2 * np.pi / 3, np.pi, 4 * np.pi / 3]
    pts = np.stack([size * np.cos(angles), size * np.sin(angles)], axis=-1)
    center_patch = Polygon(pts, zorder=10.0, **kwargs)
    center_patch.set_transform(transform)

    ax.add_patch(patch)
    ax.add_patch(center_patch)


def draw_history(
    ax: Axes,
    agent_type: AgentType,
    agent_history: StateTensor,
    agent_extent: Tensor,
    agent_to_world_tf: Tensor,
    start_alpha: float = 0.2,
    end_alpha: float = 0.5,
    **kwargs,
):
    T = agent_history.shape[0]
    alphas = np.linspace(start_alpha, end_alpha, T)
    for t in range(T):
        draw_agent(
            ax,
            agent_type,
            agent_history[t],
            agent_extent,
            agent_to_world_tf,
            alpha=alphas[t],
            **kwargs,
        )


def draw_map(
    ax: Axes, map: Tensor, base_frame_from_map_tf: Tensor, alpha=1.0, **kwargs
):
    patch_size: int = map.shape[-1]
    map_array = RasterizedMap.to_img(map.cpu())
    
    # H x W x 3
    # brightened_map_array = map_array * 0.0 + 1.0
    brightened_map_array = map_array * 0.2 + 0.8
    
    # make background white
    # print(map_array.shape)
    # background = np.all(brightened_map_array == 0.8, axis=2)
    # print(background.shape)
    # brightened_map_array[background] = 1.0

    im = ax.imshow(
        brightened_map_array,
        extent=[0, patch_size, patch_size, 0],
        clip_on=True,
        **kwargs,
    )
    transform = (
        mtransforms.Affine2D(matrix=base_frame_from_map_tf.cpu().numpy()) + ax.transData
    )
    im.set_transform(transform)

    coords = np.array(
        [[0, 0, 1], [patch_size, 0, 1], [patch_size, patch_size, 1], [0, patch_size, 1]]
    )
    world_frame_corners = base_frame_from_map_tf.cpu().numpy() @ coords[:, :, None]
    xmin = np.min(world_frame_corners[:, 0, 0])
    xmax = np.max(world_frame_corners[:, 0, 0])
    ymin = np.min(world_frame_corners[:, 1, 0])
    ymax = np.max(world_frame_corners[:, 1, 0])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_scene_batch(
    batch: SceneBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    show: bool = True,
    legend: bool = True,
    close: bool = True,
    controlled_names: Optional[str] = None,
    lim_scale = 1.0,
    show_gt = False,
    agent_name_to_color={},
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    colors = list(mcolors.TABLEAU_COLORS)

    num_agents: int = batch.num_agents[batch_idx].item()

    agent_from_world_tf: Tensor = batch.centered_agent_from_world_tf[batch_idx].cpu()

    if batch.maps is not None:
        centered_agent_id = 0
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx, centered_agent_id].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        draw_map(
            ax,
            batch.maps[batch_idx, centered_agent_id],
            agent_from_raster_tf,
            alpha=1.0,
        )

    base_frame_from_agent_tf = torch.eye(3)
    agent_hist = batch.agent_hist[batch_idx]
    agent_type = batch.agent_type[batch_idx]
    agent_extent = batch.agent_hist_extent[batch_idx, :, -1]
    agent_fut = batch.agent_fut[batch_idx]
    
    color_num = 6
    palette = sns.color_palette("husl", color_num+1)
    palette = [palette[i] for i in range(7) if i != 2]

    for agent_id in range(num_agents):
        name = batch.agent_names[batch_idx][agent_id]
        
        if controlled_names is not None and batch.agent_names[batch_idx][agent_id] in controlled_names:
            control_id = controlled_names.index(name)
            color = palette[control_id % color_num]
            # color = palette[0]
            alpha = 1.0
        else:
            # color = 'olive'
            color = (46/255., 141/255., 30/255.)
            alpha = 0.7
        
        if name not in agent_name_to_color:
          agent_name_to_color[name] = color
        else:
          color = agent_name_to_color[name]

        if agent_hist[agent_id, -1].isnan().any():
            continue

        # ax.plot(
        #     agent_hist.get_attr("x")[agent_id],
        #     agent_hist.get_attr("y")[agent_id],
        #     # c="orange",
        #     c=color,
        #     ls="--",
        #     label="Agent History" if agent_id == 0 else None,
        # )
        draw_agent(
            ax,
            agent_type[agent_id],
            agent_hist[agent_id, -1],
            agent_extent[agent_id],
            base_frame_from_agent_tf,
            facecolor=color,
            # facecolor='olive',
            edgecolor="k",
            alpha=alpha,
            label="Agent Current" if agent_id == 0 else None,
        )
        if show_gt:
            ax.plot(
                agent_fut.get_attr("x")[agent_id].cpu(),
                agent_fut.get_attr("y")[agent_id].cpu(),
                c="darkgreen",
                label="Agent Future" if agent_id == 0 else None,
            )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.plot(
            batch.robot_fut.get_attr("x")[batch_idx, 1:],
            batch.robot_fut.get_attr("y")[batch_idx, 1:],
            label="Ego Future",
            c="blue",
        )
        ax.scatter(
            batch.robot_fut.get_attr("x")[batch_idx, 0],
            batch.robot_fut.get_attr("y")[batch_idx, 0],
            s=20,
            c="blue",
            label="Ego Current",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")
    if legend:
        ax.legend(loc="best", frameon=True)

    full_meter = 100

    x_start_ratio = 0.2
    y_start_ratio = 0.5

    x_start = -lim_scale * full_meter * x_start_ratio
    y_start = -lim_scale * full_meter * y_start_ratio

    x_end = lim_scale * full_meter * (1 - x_start_ratio)
    y_end = lim_scale * full_meter * (1 - y_start_ratio)


    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)

    # Doing this because the imshow above makes the map origin at the top.
    ax.invert_yaxis()

    if show:
        plt.show()

    if close:
        plt.close()

    return ax, agent_name_to_color


import numpy as np
import torch
import copy
from trajdata.maps.vec_map_elements import Polyline
from simplification.cutil import simplify_coords
from enum import IntEnum

from trajdata.utils.arr_utils import transform_coords_np, rotation_matrix

class RoadLaneType(IntEnum):
  CENTER = 2
  LEFT_EDGE = 6
  RIGHT_EDGE = 6

def extract_lane_vecs(vec_map, center_in_world_xyzh, map_range, lanes=None, og_format: bool = False):
  center_in_world_xyz = center_in_world_xyzh[:3]
  center_in_world_h = -center_in_world_xyzh[-1]

  center_from_world_tf = np.zeros((3, 3))
  center_from_world_tf[:2, :2] = rotation_matrix(center_in_world_h)
  center_from_world_tf[2, 2] = 1.0

  if lanes is not None:
    close_lanes = lanes
  else:
    lane_dist = np.sqrt(2.0) * map_range

    close_lanes = vec_map.get_lanes_within(center_in_world_xyz, lane_dist)

  data_vecs = {'center': [], 'left_edge': [], 'right_edge': []}

  vertex_dists = []
  for lane in close_lanes:
    center_points = Polyline(simplify_coords(lane.center.points[..., :2], 0.1)).interpolate(max_dist=5).points[..., :2]
    center_cnt = len(center_points) * 4

    vertex_dists.append(np.linalg.norm(center_points[1:] - center_points[:-1], axis=-1))

    if lane.left_edge is None:
        left_edge_points = None
    else:    
        left_edge_points = lane.left_edge.interpolate(num_pts=center_cnt).points[..., :2]
        # left_edge_points = lane.left_edge.points[..., :2]
    
    if lane.right_edge is None:
        right_edge_points = None
    else:
        right_edge_points = lane.right_edge.interpolate(num_pts=center_cnt).points[..., :2]
        # right_edge_points = lane.right_edge.points[..., :2]

    points = {'center': center_points, 'left_edge': left_edge_points, 'right_edge': right_edge_points}

    tls = 0

    for k, v in points.items():
      if v is None:
          continue

      line_type = float(RoadLaneType[k.upper()])

      v = v.copy() - center_in_world_xyz[:2]
      v = transform_coords_np(v, center_from_world_tf, translate=False)

      dist_mask = np.linalg.norm(v, axis=-1) < map_range

      v = v[dist_mask]
      if v.shape[0] < 2:
          continue
      
      vector = np.zeros((v.shape[0]-1, 6))
      vector[:, 0:2] = v[:-1, :2]
      vector[:, 2:4] = v[1:, :2]
      vector[:, 4] = line_type
      vector[:, 5] = tls
      
      data_vecs[k].append(vector)
  
  if not og_format:
    for k, v in data_vecs.items():
      data_vecs[k] = np.concatenate(v, axis=0)
    
    data_vecs["query_pt"] = np.repeat(np.expand_dims(center_in_world_xyzh, axis=0), data_vecs["center"].shape[0], axis=0)
  
  return data_vecs


def process_lines(line_vec, max_num):
  line_mask = np.zeros(max_num).astype(bool)
  dist = line_vec[..., 0] ** 2 + line_vec[..., 1] ** 2
  idx = np.argsort(dist)
  line_vec = line_vec[idx]

  lane_num = min(max_num, line_vec.shape[0])
  
  line_vec = line_vec[:max_num]
  line_vec = np.pad(line_vec, ((0, max_num - lane_num), (0, 0)))
  line_mask[:lane_num] = True

  return line_vec, line_mask

def process_lctgen_map_inp(data_vecs, map_range):
  center_num = 384
  bound_num = 128
  edge_num = 192
  cross_num = 32

  center_lines = np.concatenate(data_vecs['center'], axis=0)
  center_lines, center_mask = process_lines(center_lines, center_num)

  left_edge_lines = np.concatenate(data_vecs['left_edge'], axis=0)
  right_edge_lines = np.concatenate(data_vecs['right_edge'], axis=0)
  edge_lines = np.concatenate([left_edge_lines, right_edge_lines], axis=0)
  edge_lines, edge_mask = process_lines(edge_lines, edge_num)

  # TODO (shuhan): can we also add points for cross / road boundary?
  cross_lines, cross_mask = np.zeros((cross_num, 6)), np.zeros(cross_num).astype(bool)
  bound_lines, bound_mask = np.zeros((bound_num, 6)), np.zeros(bound_num).astype(bool)

  center = copy.deepcopy(center_lines)
  center[..., :4] /= map_range
  edge = copy.deepcopy(edge_lines)
  edge[..., :4] /= map_range

  lane_inp = np.concatenate([center, edge, cross_lines, bound_lines], axis=0)
  lane_mask = np.concatenate([center_mask, edge_mask, cross_mask, bound_mask], axis=0)


  return lane_inp, lane_mask, center_lines, center_mask, edge_lines, edge_mask

from prosim.dataset.data_utils import rotate
def plot_full_map(batch, lim_scale, controlled_names, vis_vecs, ax=None, show_gt=False, agent_name_to_color={}):
  ax, agent_name_to_color = plot_scene_batch(batch, 0, show=False, close=False, lim_scale=lim_scale, legend=False, controlled_names=controlled_names, ax=ax, show_gt=show_gt, agent_name_to_color=agent_name_to_color)

  center_in_world_xyzh = batch.centered_agent_state.as_format('x,y,z,h').cpu().numpy()[0]

  # vec_map = batch.vector_maps[0]
  # vecs = extract_lane_vecs(vec_map, center_in_world_xyzh, 150)
  center = vis_vecs['center']

  for j in range(center.shape[0]):
    x0, y0, x1, y1, = center[j, :4]
    if x0 == 0: break
    ax.plot((x0, x1), (y0, y1), '--', color='gray', linewidth=0.8, alpha=0.5)

  # polyline_tensors_local = batch.extras['road_edge_polyline'][0]
  # for lane in polyline_tensors_local:
  #   ax.plot(lane[:, 0].cpu(), lane[:, 1].cpu(), color='gray', linewidth=1.5, alpha=0.8)
  
  return ax, agent_name_to_color

def plot_batch_prompts(batch, ax, agent_name_to_color, controlled_names, font_size=10.0, show_all_name=False, all_text_pos={}):
  if 'goal' in batch.extras['condition'].keys():
    goal_nidx = batch.extras['condition']['goal']['prompt_idx'][0, :, 0].cpu().numpy().tolist()
    goal_cond_input = batch.extras['condition']['goal']['input'][0, :, :2].cpu().numpy()
    goal_cond_mask = batch.extras['condition']['goal']['mask'][0, :].cpu().numpy()

  if 'drag_point' in batch.extras['condition'].keys():
    drag_point_nidx = batch.extras['condition']['drag_point']['prompt_idx'][0, :, 0].cpu().numpy().tolist()
    drag_point_input = batch.extras['condition']['drag_point']['input'][0, :, :, :2].cpu().numpy()
    drag_point_mask = batch.extras['condition']['drag_point']['mask'][0, :].cpu().numpy()

  all_init_pos = batch.extras['prompt']['motion_pred']['position'].cpu().numpy()
  all_init_heading = batch.extras['prompt']['motion_pred']['heading'].cpu().numpy()

  for nidx, name in enumerate(batch.extras['prompt']['motion_pred']['agent_ids'][0]):
    color = agent_name_to_color[name]

    is_prompted = name in controlled_names or show_all_name
    plot_prompt = name in controlled_names

    if is_prompted:
      init_pos = all_init_pos[0, nidx, :2]
      init_heading = all_init_heading[0, nidx]

      agent_name = f'A{nidx}'
      text_pos = all_text_pos.get(nidx, [0, 1])

      if plot_prompt:
        if text_pos[0] == -1:
            pass
        if text_pos[0] == 0: # up
            ax.text(init_pos[0], init_pos[1]-1.5, agent_name, fontsize=font_size, color=color, ha='center', va='bottom', fontweight=800, clip_on=True)
        elif text_pos[0] == 1: # down
            if nidx == 8:
              ax.text(init_pos[0], init_pos[1]+1.0, agent_name, fontsize=font_size, color=color, ha='center', va='top', fontweight=800, clip_on=True)
            else:
              ax.text(init_pos[0], init_pos[1]+1.5, agent_name, fontsize=font_size, color=color, ha='center', va='top', fontweight=800, clip_on=True)
        elif text_pos[0] == 2: # right
            if nidx == 23:
              ax.text(init_pos[0]+3.0, init_pos[1]-1.5, agent_name, fontsize=font_size, color=color, ha='left', va='center', fontweight=800, clip_on=True)
            else:
              ax.text(init_pos[0]+3.0, init_pos[1], agent_name, fontsize=font_size, color=color, ha='left', va='center', fontweight=800, clip_on=True)
        elif text_pos[0] == 3: # left
            ax.text(init_pos[0]-3.0, init_pos[1], agent_name, fontsize=font_size, color=color, ha='right', va='center', fontweight=800, clip_on=True)
      else:
        ax.text(init_pos[0], init_pos[1]-1.5, agent_name, fontsize=8.0, color='r', ha='center', va='bottom', fontweight=300, clip_on=True)
    
      if 'goal' in batch.extras['condition'].keys() and plot_prompt:
        if nidx in goal_nidx:
          cidx = goal_nidx.index(nidx)

        if goal_cond_mask[cidx]:
          goal_pos = goal_cond_input[cidx]
          goal_pos = rotate(goal_pos[0], goal_pos[1], init_heading)[0] + init_pos

          ax.scatter(goal_pos[0], goal_pos[1], color=color, s=80, marker='x', clip_on=True)

          if text_pos[1] == -1: 
            pass
          elif text_pos[1] == 0: # up
            ax.text(goal_pos[0], goal_pos[1]-2.0, agent_name, fontsize=font_size, color=color, ha='center', va='bottom', fontweight=800, clip_on=True)
          elif text_pos[1] == 1: # down
            ax.text(goal_pos[0], goal_pos[1]+2.0, agent_name, fontsize=font_size, color=color, ha='center', va='top', fontweight=800, clip_on=True)
          elif text_pos[1] == 2: # right
            ax.text(goal_pos[0]+3.5, goal_pos[1], agent_name, fontsize=font_size, color=color, ha='left', va='center', fontweight=800, clip_on=True)
          elif text_pos[1] == 3: # left
            ax.text(goal_pos[0]-3.5, goal_pos[1], agent_name, fontsize=font_size, color=color, ha='right', va='center', fontweight=800, clip_on=True)


      if 'drag_point' in batch.extras['condition'].keys() and plot_prompt:
        if nidx in drag_point_nidx:
          cidx = drag_point_nidx.index(nidx)

          if drag_point_mask[cidx]:
            drag_pos = drag_point_input[cidx]
            drag_pos = rotate(drag_pos[..., 0], drag_pos[..., 1], init_heading) + init_pos
            ax.scatter(drag_pos[..., 0], drag_pos[..., 1], color=color, s=15, marker='o', clip_on=True)
  
  return ax

def plot_model_output(batch, ax, agent_name_to_color, controlled_names, output):
    vis_indices = []
    for idx, name in enumerate(output['pair_names']):
      batch_id, _, T = name.split('-')
      if batch_id == '0' and int(T) == 0:
        vis_indices.append(idx)
    
    batch_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][0]
    
    for vidx in vis_indices:
      name = output['pair_names'][vidx]
    
      aname = name.split('-')[1]
      color = agent_name_to_color[aname]
    
      if aname not in controlled_names:
        alpha = 0.6
      else:
        alpha = 0.8
    
      rollout_id = '-'.join(name.split('-')[:2])
      roll_traj = output['rollout_trajs'][rollout_id]['traj'].cpu().detach().numpy()
      init_pos = output['rollout_trajs'][rollout_id]['init_pos'].cpu().detach().numpy()
      init_heading = output['rollout_trajs'][rollout_id]['init_heading'].cpu().detach().numpy()
    
      roll_traj = rotate(roll_traj[..., 0], roll_traj[..., 1], init_heading) + init_pos
      # ax.scatter(roll_traj[:, 0], roll_traj[:, 1], color=color, alpha=alpha, s=10, marker='x')
      ax.plot(roll_traj[:, 0], roll_traj[:, 1], linestyle='-', color=color, alpha=alpha, linewidth=2.0)

    return ax


from matplotlib import pyplot as plt

def plot_demo_fig(batch, controlled_names, output, vis_vecs, scale=1.00, hide_axis=True, save_path=None, show_all_name=False, show_gt=False, all_text_pos={}, agent_name_to_color={}):
    fig, ax = plt.subplots()

    ax, agent_name_to_color = plot_full_map(batch, scale, controlled_names, vis_vecs, ax, show_gt, agent_name_to_color)

    ax = plot_batch_prompts(batch, ax, agent_name_to_color, controlled_names, font_size=10.0, show_all_name=show_all_name, all_text_pos=all_text_pos)

    if output is not None:
        ax = plot_model_output(batch, ax, agent_name_to_color, controlled_names, output)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    if hide_axis:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    return fig, ax