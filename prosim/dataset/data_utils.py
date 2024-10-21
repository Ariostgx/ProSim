import time
import os
import torch
import numpy as np
from pathlib import Path
import json

from torch.nn.utils.rnn import pad_sequence
from typing import List
from trajdata import AgentType
from trajdata.maps.map_api import MapAPI
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.utils.arr_utils import transform_coords_np
from trajdata.utils.state_utils import transform_state_np_2d
from trajdata.utils.state_utils import StateArray
from trajdata.utils.arr_utils import rotation_matrix, angle_wrap

from prosim.dataset.motion_tag_utils import MotionTags, exclusion_groups, priority_dict, integrate_motion_tags, remove_short_motion_tags, resolve_and_adjust_conflicts

from enum import IntEnum

NUSC_TS = 0.5
class RoadLaneType(IntEnum):
    CENTER = 1
    LEFT_EDGE = 2
    RIGHT_EDGE = 3

default_trajdata_cfg = {
    "centric":"agent",
    "only_types":[AgentType.VEHICLE],
    "state_format":"x,y,z,xd,yd,xdd,ydd,h",
    "obs_format":"x,y,z,xd,yd,xdd,ydd,s,c",
    "incl_robot_future":False,
    "incl_raster_map":True,
    "raster_map_params":{
        "px_per_m": 2,
        "map_size_px": 224,
        "offset_frac_xy": (-0.5, 0.0),
        "num_workers": 0,
    },
    "vector_map_params":{
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                "collate": True,
                "keep_in_memory": False,
    },
    "incl_vector_map":True,
    "verbose":True,
    "standardize_data":True,
    "standardize_derivatives": False,
    "use_all_agents":False,
    }

def rotate(x, y, angle):
  if isinstance(x, torch.Tensor):
      other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
      other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
      output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

  else:
      other_x_trans = np.cos(angle) * x - np.sin(angle) * y
      other_y_trans = np.cos(angle) * y + np.sin(angle) * x
      output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
  return output_coords

class VecLanes:
    def __init__(self, vec_lanes):
        self.vec_lanes = vec_lanes
    
    def __to__(self, device, non_blocking=False):
        for vec_lane in self.vec_lanes:
            vec_lane = vec_lane.to(device, non_blocking=non_blocking)
        return self

    def __collate__(self, batch):
        result = []
        for item in batch:
            result += item.vec_lanes
        
        return VecLanes(result)

    def __getitem__(self, idx):
        return self.vec_lanes[idx]


def get_vectorized_lanes(batch_ele, data_cfg, map_range):
    scene_id = batch_ele.scene_id
    env_name = batch_ele.cache.scene.env_name
    map_cfg = data_cfg.MAP

    cached_tensor = _get_vectorized_lanes_from_cache(data_cfg, scene_id, env_name)

    vec_map = batch_ele.vec_map
    if vec_map is None:
        map_api = MapAPI(Path(data_cfg.CACHE_PATH))
        vec_map = map_api.get_map(
            batch_ele.map_name,
            batch_ele.cache if batch_ele.cache.is_traffic_light_data_cached() else None,
            **default_trajdata_cfg['vector_map_params'],
        )
        batch_ele.vec_map = vec_map

    if cached_tensor is not None:
        return VecLanes([cached_tensor])

    if type(batch_ele) == AgentBatchElement:
        agent_init_state = batch_ele.agent_history_np[-1]
        agent_from_world_tf = batch_ele.agent_from_world_tf
        world_from_agent_tf = np.linalg.inv(agent_from_world_tf)
        agent_in_world = transform_state_np_2d(agent_init_state, world_from_agent_tf)
    else:
        agent_in_world = batch_ele.centered_agent_state_np
        agent_from_world_tf = batch_ele.centered_agent_from_world_tf
    
    
    agent_in_world_xyz = agent_in_world.as_format('x,y,z')

    scene_ts = batch_ele.scene_ts

    result = _get_vectorized_lanes_from_vector_map(agent_in_world_xyz, agent_from_world_tf, vec_map, scene_ts, map_cfg, map_range, env_name)

    return result

def _get_vectorized_lanes_from_cache(config, scene_id, dataset_name):
    save_root = config.DATA_PATHS.VECTOR_LANE_CACHE
    if save_root:
        cache_path = os.path.join(save_root, dataset_name, scene_id + '.pt')
        if os.path.exists(cache_path):
            vector_lane_tensor = torch.load(cache_path)
            # print('\tloading lane from cache: ', cache_path)
            return vector_lane_tensor
    return None


def _get_vectorized_lanes_from_vector_map(agent_in_world_xyz, agent_from_world_tf, vec_map, scene_ts, map_cfg, map_range, env_name):
    # obtain vectorized lane data for the full scene
    # extracing map from a large area around the agent
    center_sample_rate = map_cfg.CENTER_SAMPLE_RATE
    edge_sample_rate = map_cfg.EDGE_SAMPLE_RATE

    collate_mode = map_cfg.COLLATE_MODE

    if collate_mode == 'type':
        data_vecs = {k: [] for k in map_cfg.INCLUDE_TYPES}
    elif collate_mode == 'lane':
        max_lane_points = map_cfg.MAX_LANE_POINTS
        data_vecs = []


    lane_dist = np.sqrt(map_range ** 2 + map_range ** 2)
    close_lanes = vec_map.get_lanes_within(agent_in_world_xyz, lane_dist)

    for lane in close_lanes:
        center_points = lane.center.points[..., :2]
        center_cnt = len(center_points)

        if lane.left_edge is None:
            left_edge_points = None
        else:    
            if 'nusc' in env_name:
                left_edge_points = lane.left_edge.interpolate(num_pts=center_cnt).points[..., :2]
            else:
                left_edge_points = lane.left_edge.points[..., :2]
        
        if lane.right_edge is None:
            right_edge_points = None
        else:
            if 'nusc' in env_name:
                right_edge_points = lane.right_edge.interpolate(num_pts=center_cnt).points[..., :2]
            else:
                right_edge_points = lane.right_edge.points[..., :2]

        points = {'center': center_points, 'left_edge': left_edge_points, 'right_edge': right_edge_points}
        tls = float(vec_map.get_traffic_light_status(lane.id, scene_ts))

        for k, v in points.items():
            if k not in map_cfg.INCLUDE_TYPES:
                continue

            if v is None:
                continue

            line_type = float(RoadLaneType[k.upper()])

            if 'edge' in k:
                sample_rate = edge_sample_rate
            else:
                sample_rate = center_sample_rate

            if v.shape[0] > sample_rate:
                v = v[::sample_rate]
            
            v = transform_coords_np(v, agent_from_world_tf)

            dist_mask = ((abs(v[:, :1]) < map_range) * (abs(v[:, 1:2]) < map_range)).squeeze()

            v = v[dist_mask]
            point_cnt = v.shape[0]
            
            if point_cnt < 2:
                continue
                
            if collate_mode == 'lane' and point_cnt > max_lane_points:
                chunk_idx = list(np.arange(0, point_cnt, max_lane_points, dtype=int))
                if chunk_idx[-1] != point_cnt:
                    chunk_idx.append(point_cnt)
            else:
                chunk_idx = [0, len(v)-1]

            for i in range(len(chunk_idx)-1):
                v_chunk = v[chunk_idx[i]:chunk_idx[i+1]]
                v_len = len(v_chunk)-1
                
                if v_len < 1:
                    continue
                
                vec_shape = max_lane_points-1 if collate_mode == 'lane' else v_len

                vector = np.zeros((vec_shape, 6))
                vector[:v_len, 0:2] = v_chunk[:-1, :2]
                vector[:v_len, 2:4] = v_chunk[1:, :2]
                vector[:v_len, 4] = line_type
                vector[:v_len, 5] = tls
                
                # invalid points for each chunk
                vector[v_len:, 4] = 0
            
                if collate_mode == 'type':
                    data_vecs[k].append(vector)
                elif collate_mode == 'lane':
                    data_vecs.append(vector)

    if collate_mode == 'lane':
        if len(data_vecs) == 0:
            return VecLanes([torch.zeros(1, 39, 6).float()])

        return VecLanes([torch.tensor(np.stack(data_vecs, axis=0)).float()])

    elif collate_mode == 'type':
        input_vecs = []
        for k, v in data_vecs.items():
            if len(v) == 0:
                continue
            else:
                v = np.concatenate(v, axis=0)
            input_vecs.append(v)
        
        if len(input_vecs) == 0:
            return VecLanes([torch.zeros(1, 6).float()])

        return VecLanes([torch.tensor(np.concatenate(input_vecs, axis=0)).float()])

def transform_coords_2d_np_offset_rot(
    coords: np.ndarray,
    offset: np.ndarray = None,
    rot_mat: np.ndarray= None,
) -> np.ndarray:
    """
    Do translation first and then rotation
    """
    if offset is not None:
        coords += offset

    if rot_mat is not None:
        coords = np.einsum("...ij,...j->...i", rot_mat, coords)

    return coords

def transform_to_frame_offset_rot(state: StateArray, frame_state: StateArray,) -> StateArray:
    """
    Returns state with coordinates relative to a frame with state frame_state.
    Does not modify state in place.

    Do translation first and then rotation

    Args:
        state (StateArray): state to transform in world coordinates
        frame_state (StateArray): state of frame in world coordinates
        rot_mat Optional[nd.array]: rotation matrix A such that c = A @ b returns coordinates in the new frame
            if not given, it is computed frome frame_state
    """
    new_state = state.copy()
    attributes = state._format_dict.keys()

    frame_heading = frame_state.heading[..., 0]
    rot_mat = rotation_matrix(-frame_heading)

    if "x" in attributes and "y" in attributes:
        # transform xy position with translation and rotation
        new_state.position = transform_coords_2d_np_offset_rot(
            state.position, offset=-frame_state.position, rot_mat=rot_mat
        )
    if "xd" in attributes and "yd" in attributes:
        # do not offset velocities, only rotate
        new_state.velocity = transform_coords_2d_np_offset_rot(
            state.velocity, rot_mat=rot_mat
        )
    if "xdd" in attributes and "ydd" in attributes:
        # do not offset velocities, only rotate
        new_state.acceleration = transform_coords_2d_np_offset_rot(
            state.acceleration, rot_mat=rot_mat
        )
    if "c" in attributes and "s" in attributes:
        new_state.heading_vector = transform_coords_2d_np_offset_rot(
            state.heading_vector, rot_mat=rot_mat
        )
    if "h" in attributes:
        new_state.heading = angle_wrap(state.heading - frame_heading)

    return new_state

def remove_parked(element: SceneBatchElement):
    is_parked = np.array([meta_dict['is_stationary'] for meta_dict in element.agent_meta_dicts])
    get_filtered_list = lambda all_idx: [idx for idx in all_idx if not is_parked[idx]]
    element.tgt_agent_idx = get_filtered_list(element.tgt_agent_idx)
    return element

def use_all_target(element: SceneBatchElement):
    return element

def get_waymo_file_template(config, dataset_name):
  source_data = dataset_name

  data_path = config.DATA_PATHS[source_data.upper()]
  split = source_data.split('_')[-1]
  if split == 'val':
    waymo_split = 'validation_splitted'
  elif split == 'test':
    waymo_split = 'testing_splitted'
  elif split == 'train':
    waymo_split = 'training_splitted'

  scene_template = os.path.join(data_path, waymo_split, waymo_split + '_{}.tfrecords')

  return scene_template

def get_waymo_scene_object(scene_name, scene_template):
    return None

class RoadEdgePolylines:
    def __init__(self, b_polyline_tensor):
        # [B, P, S, 2]
        self.b_polyline_tensor = b_polyline_tensor
    
    def __to__(self, device, non_blocking=False):
        self.b_polyline_tensor = self.b_polyline_tensor.to(device, non_blocking=non_blocking)
        return self

    def __collate__(self, batch):
        device = batch[0].b_polyline_tensor.device
        polyline_tensors = [item.b_polyline_tensor for item in batch]
        B = len(batch)
        P = max([p_tensor.shape[1] for p_tensor in polyline_tensors]) # number of polylines
        S = max([p_tensor.shape[2] for p_tensor in polyline_tensors]) # number of points per polyline
        b_polyline = torch.ones(B, P, S, 2, device=device) * torch.nan

        for bidx in range(B):
            p_tensor = polyline_tensors[bidx]
            b_polyline[bidx, :p_tensor.shape[1], :p_tensor.shape[2]] = p_tensor[0]
        
        return RoadEdgePolylines(b_polyline)

    def __getitem__(self, idx):
        return self.b_polyline_tensor[idx]

def _segment_polylines(polyline_list):
    MAX_POLYLINE_POINTS = 20
    segmented_polylines = []
    for polyline in polyline_list:
        if len(polyline) > MAX_POLYLINE_POINTS:
            chunk_idx = list(np.arange(0, len(polyline), MAX_POLYLINE_POINTS, dtype=int))
            if chunk_idx[-1] != len(polyline):
                chunk_idx.append(len(polyline))
            for i in range(len(chunk_idx)-1):
                segmented_polylines.append(polyline[chunk_idx[i]:chunk_idx[i+1]])
        else:
            segmented_polylines.append(polyline)
    return segmented_polylines

def _transform_poly_to_local(road_edge_nps, agent_from_world_tf):
    MIN_POLYLINE_POINTS = 2 * 3 + 1
    road_edge_tensors = [torch.tensor(polyline).float() for polyline in road_edge_nps if len(polyline) >= MIN_POLYLINE_POINTS]

    # [P, S, 2]
    # P: number of polylines
    # S: number of points in each polyline
    road_edge_tensors = pad_sequence(road_edge_tensors, batch_first=True, padding_value=torch.nan)

    road_edge_local = transform_coords_np(road_edge_tensors.numpy(), agent_from_world_tf)
    
    # [1, P, S, 2]: torch tensor
    road_edge_local = torch.tensor(road_edge_local).float()[None, :]
    
    return road_edge_local

def _load_road_edge_from_waymo(scene_id, scene_template):
    
    waymo_scene = get_waymo_scene_object(scene_id, scene_template)
    
    road_edge_nps = []
    for map_feature in waymo_scene.map_features:
        if map_feature.HasField('road_edge'):
            polyline = map_feature.road_edge.polyline
            polyline_np = np.array([[map_point.x, map_point.y] for map_point in polyline])
            road_edge_nps.append(polyline_np)

    return road_edge_nps


def get_waymo_road_edges(batch_ele, config, split):
    
    scene_id = batch_ele.scene_id
    dataset_name = config.SOURCE[split][0]

    save_root = config.DATA_PATHS.WAYMO_ROAD_EDGE_CACHE
    if save_root:
        save_root = os.path.join(save_root, dataset_name)
        road_edge_path = os.path.join(save_root, scene_id + '.npy')
        if os.path.exists(road_edge_path):
            road_edge_nps = np.load(road_edge_path, allow_pickle=True)
            polyline_local_list = _segment_polylines(road_edge_nps)
            polyline_local_tensor = _transform_poly_to_local(polyline_local_list, batch_ele.centered_agent_from_world_tf)
            return RoadEdgePolylines(polyline_local_tensor)

    scene_template = get_waymo_file_template(config, dataset_name)
    road_edge_nps = _load_road_edge_from_waymo(scene_id, scene_template)
    agent_from_world_tf = batch_ele.centered_agent_from_world_tf
    polyline_local_list = _segment_polylines(road_edge_nps)
    polyline_local_tensor = _transform_poly_to_local(polyline_local_list, agent_from_world_tf)
    return RoadEdgePolylines(polyline_local_tensor)

def get_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2

    # Check for overlap
    if start1 <= end2 and end1 >= start2:
        # Calculate overlapping section
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return [overlap_start, overlap_end]
    else:
        return None  # No overlap


def filter_scene_tags(scene_id_tags: List[str], scene_interval: List[int], tgt_tag_types: List[str], trajdata_ts: float, use_waymo: bool):
    '''
    Filter and structure tags based on their type, interval, and agent properties within a specified scene interval. 
    Handles a list of tag strings and categorizes them based on the provided target tag types.
    
    Input:
      scene_id_tags: List[str]
        - A list of tag strings, each representing a specific tag in the scene. Each tag string includes tag type, agent information, and interval.
      scene_interval: List[int]
        - A list of two integers representing the start and end timestamps of the scene interval. Used to filter tags based on their temporal overlap with this interval.
      tgt_tag_types: List[str]
        - A list of strings representing the target tag types to filter. Tags not matching these types are excluded from the output.
    
    Output:
      filtered_tags: List[dict]
        - A list of dictionaries, each representing a filtered tag. The dictionary includes:
          - 'agents': List[str], a list of agent identifiers involved in the tag.
          - 'interval': Tuple[int, int], the interval of the tag overlapping with the scene interval.
          - 'type': str, the type of the tag.
    '''
    
    filtered_tags = []
    
    for tag_str in scene_id_tags:
        tag_type = tag_str.split('(')[0].split('Temporal')[0]
        if tag_type not in tgt_tag_types:
            continue
    
        tag_s, tag_e = tag_str.split('at ')[-1].split(')')[0].split('-')
        
        if use_waymo:
            ts_ratio = 1
        else:
            ts_ratio = int(NUSC_TS / trajdata_ts)
        
        tag_s, tag_e = int(tag_s) * ts_ratio, int(tag_e) * ts_ratio
    
        overlap = get_overlap((tag_s, tag_e), scene_interval)
        if overlap is None:
            continue

        overlap[0] -= scene_interval[0]
        overlap[1] -= scene_interval[0]
        
        is_binary = ',' in tag_str
    
        if is_binary:
            agents = tag_str.split('(')[-1].split(' at')[0].split(', ')
        else:
            agents = [tag_str.split('(')[-1].split(' at')[0]]

        arg_type = 'binary' if is_binary else 'unary'
    
        tag = {'agents': agents, 'interval': overlap, 'tag': tag_type, 'type': arg_type}
        filtered_tags.append(tag)

    return MotionTags([filtered_tags])

def get_scene_motion_tag(batch_ele, config, all_motion_tags, split):
    scene_id = batch_ele.scene_id

    if all_motion_tags is None:
        motion_tag_path = config.DATASET.DATA_PATHS.MOTION_TAGS[split.upper()]
        
        scene_idx = int(scene_id.split('_')[-1])
        scene_num = scene_idx % 100
        
        scene_file = scene_id + '.json'
        scene_file = os.path.join(motion_tag_path, str(scene_num), scene_file)

        if os.path.exists(scene_file):
            with open(scene_file, 'r') as f:
                scene_id_tags = json.load(f)
        else:
            return MotionTags([[]])

    else:
        if scene_id not in all_motion_tags:
            return MotionTags([[]])

        scene_id_tags = all_motion_tags[scene_id]

    scene_dt = batch_ele.dt
    scene_len = max(batch_ele.agent_future_lens_np)
    
    scene_start_ts = batch_ele.scene_ts
    scene_end_ts = batch_ele.scene_ts + scene_len
    scene_interval = (scene_start_ts, scene_end_ts)

    tgt_tag_types = config.PROMPT.CONDITION.MOTION_TAG.USED_TAGS

    use_waymo = 'waymo' in config.DATASET.DATA_PATHS.MOTION_TAGS[split.upper()]
    
    scene_tags = filter_scene_tags(scene_id_tags, scene_interval, tgt_tag_types, scene_dt, use_waymo)

    if config.PROMPT.CONDITION.MOTION_TAG.USE_PROCESSING:
        input_tags = {'result': scene_tags}

        # Example of using the function with the tolerance set to 5 timestamps
        result_tags = integrate_motion_tags(input_tags, tolerance=config.PROMPT.CONDITION.MOTION_TAG.INTEGRATE_TOLERANCE)
        result_tags = remove_short_motion_tags(result_tags, min_duration=config.PROMPT.CONDITION.MOTION_TAG.MIN_DURATION)
        result_tags = resolve_and_adjust_conflicts(result_tags, exclusion_groups, priority_dict)

        return result_tags['result']
    else:
        return scene_tags

import re

class LLMTexts:
    def __init__(self, llm_texts):
        # List[Dict]
        self.llm_texts = llm_texts
    
    def __to__(self, device, non_blocking=False):
        return self

    def __collate__(self, batch):
        result = []
        for item in batch:
            result += item.llm_texts

        return LLMTexts(result)

    def __getitem__(self, idx):
        if idx < len(self.llm_texts):
            return self.llm_texts[idx]
        else:
            return []

    def __len__(self):
        return len(self.llm_texts[0])

def load_text_file(txt_file):
    snapshot_id = txt_file.split('/')[-1].split('.')[0]
    scene_id = '_'.join(snapshot_id.split('_')[:2])
    
    with open(txt_file, 'r') as f:
        raw_lines = f.readlines()
        cleaned_lines = process_lines(raw_lines)  # Assume process_lines is defined elsewhere
    return scene_id, snapshot_id, cleaned_lines


def process_lines(lines):
  # Remove the line number and extra space
  lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines]

  # Remove the first introduction line with "Here are"
  lines = [re.sub(r'Here are.*', '', line) for line in lines]

  # Remove empty lines and extra quotes
  lines = [line for line in lines if line]
  lines = [line.replace('"', '') for line in lines]

  return lines

def get_llm_text(batch_ele, config, split):
    scene_id = batch_ele.scene_id
    llm_text_folder = config.PROMPT.CONDITION.LLM_TEXT_FOLDER[split.upper()]

    scene_id_num = int(scene_id.split('_')[-1]) % 100
    llm_text_path = os.path.join(llm_text_folder, str(scene_id_num), f'{scene_id}_10_90_output.txt')
    if os.path.exists(llm_text_path):
        try:
            _, _, cleaned_lines = load_text_file(llm_text_path)
            return LLMTexts([cleaned_lines])
        except:
            return LLMTexts([[]])
    return LLMTexts([[]])