import random

import copy
import torch
import numpy as np

from trajdata import AgentBatch
from trajdata.utils.arr_utils import rotation_matrix
from trajdata.utils.state_utils import StateTensor
from prosim.dataset.data_utils import transform_to_frame_offset_rot, transform_coords_2d_np_offset_rot
from trajdata.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch

from prosim.dataset.data_utils import rotate
from prosim.dataset.prompt_utils import AgentStatusGenerator
from prosim.dataset.condition_utils import ConditionGenerator, BatchCondition
from torch.nn.utils.rnn import pad_sequence

# Size notations: 
# B - batch size
# L - number of lanes
# N - number of agents
# D - dimension of vector (different for different vectors)
# T - number of time steps

# -----------------------------------------------------------------------------
# data objects
# -----------------------------------------------------------------------------

# need to be able to convert to device
class InputMaskData:
    def __init__(self, input, mask, position=None, heading=None, agent_ids=None):
        self.input = input
        self.mask = mask
        self.position = position
        self.heading = heading
        
        # only for hist observation
        # agent_ids is a list of agent names for each observation
        self.agent_ids = agent_ids
        
        # avoid using invalid data
        assert input[mask].isnan().any() == False
    
    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)
    
    def __to__(self, device, non_blocking=False):
        self.input = self.input.to(device, non_blocking=non_blocking)
        self.mask = self.mask.to(device, non_blocking=non_blocking)
        
        if self.position is not None:
            self.position = self.position.to(device, non_blocking=non_blocking)
            self.heading = self.heading.to(device, non_blocking=non_blocking)
        
        if self.agent_ids is not None:
            self.agent_ids = self.agent_ids

        return self

    def __setitem__(self, key, value):
        assert key in ['input', 'mask', 'position', 'heading', 'agent_ids']
        setattr(self, key, value)

    def __getitem__(self, key):
        assert key in ['input', 'mask', 'position', 'heading', 'agent_ids']
        return getattr(self, key)

    def keys(self):
        keys = ['input', 'mask']
        if self.position is not None:
            keys += ['position', 'heading']
        if self.agent_ids is not None:
            keys += ['agent_ids']
        return keys

class BatchDataDict:
    def __init__(self, input_dict):
        self.input = input_dict
    
    def __to__(self, device, non_blocking=False):
        for key in self.input.keys():
            if type(self.input[key]) == list:
                continue
            elif type(self.input[key]) == torch.Tensor:
                self.input[key] = self.input[key].to(device, non_blocking=non_blocking)
            else:
                self.input[key] = self.input[key].__to__(device, non_blocking=non_blocking)
        return self

    def __getitem__(self, key):
        return self.input[key]

    def keys(self):
        return self.input.keys()


class IOPairData:
    def __init__(self, io_pairs):
        self.io_pairs = io_pairs
    
    def __to__(self, device, non_blocking=False):
        for pair_name in self.io_pairs.keys():
            for key, value in self.io_pairs[pair_name].items():
                if type(value) == InputMaskData:
                    self.io_pairs[pair_name][key] = value.__to__(device, non_blocking=non_blocking)
                else:
                    self.io_pairs[pair_name][key] = value.to(device, non_blocking=non_blocking)

        return self

    def __getitem__(self, key):
        assert key in self.io_pairs.keys()
        return self.io_pairs[key]

    def __len__(self):
        return len(self.io_pairs)
    
    def keys(self):
        return self.io_pairs.keys()


class BatchPrompt:
    def __init__(self, all_prompts):
        self.all_prompts = all_prompts
    
    def __to__(self, device, non_blocking=False):
        for task in self.all_prompts.keys():
            for prompt_key in self.all_prompts[task].keys():
                if type(self.all_prompts[task][prompt_key]) == list:
                    continue
                self.all_prompts[task][prompt_key] = self.all_prompts[task][prompt_key].to(device, non_blocking=non_blocking)
        
        return self

    def __getitem__(self, key):
        assert key in self.all_prompts.keys()
        return self.all_prompts[key]
    
    def __len__(self):
        return len(self.all_prompts)
    
    def keys(self):
        return self.all_prompts.keys()

# -----------------------------------------------------------------------------
# init map formating functions
# -----------------------------------------------------------------------------
# output: produce initial map for each scene
# shape: [B, L, D]

def get_local_vec_map(full_vec, local_pos, map_cfg):
  if len(full_vec.shape) == 2: # (M, D)
    position = full_vec[..., :2]
  else: #(M, N, D)
    mask = full_vec[..., 4] > 0
    cnt = mask.sum(dim=1)
    cnt[cnt == 0] = 1
    position = full_vec[..., :2].sum(dim=1) / cnt[:, None] # (M, 2)

  full_dist = torch.norm(position - local_pos, dim=-1)
  dist_mask = full_dist < map_cfg.LOCAL_RANGE

  local_vec = full_vec[dist_mask]
  local_dist = full_dist[dist_mask]

  num_points_poly = local_vec.shape[1]
  local_mask = torch.zeros(map_cfg.MAX_POINTS, num_points_poly, dtype=torch.bool) # (num_polylines, num_points_each_polylines)
  p_num = min(map_cfg.MAX_POINTS, local_vec.shape[0])
  local_mask[:p_num] = local_vec[:p_num, :, 4] > 0

  if local_vec.shape[0] > map_cfg.MAX_POINTS:
    sort_idx = torch.argsort(local_dist)
    local_vec = local_vec[sort_idx[:map_cfg.MAX_POINTS]]
  else:
    # pad
    pad_size = map_cfg.MAX_POINTS - local_vec.shape[0]
    pad_shape = [pad_size] + list(local_vec.shape[1:])
    local_vec = torch.cat([local_vec, torch.zeros(pad_shape)], dim=0)
  
  return local_vec, local_mask

def local_map_to_sym_coord(local_vec):
    if len(local_vec.shape) == 2: # (M, D)
        start = local_vec[..., :2]
        end = local_vec[..., 2:4]
        
    else: #(M, N, D)
        M = local_vec.shape[0]

        mask = local_vec[..., 4] > 0
        cnt = mask.sum(dim=1)
        start = local_vec[:, 0, :2]

        i = torch.arange(M)[:, None]
        j = cnt[:, None] - 1
        end = local_vec[i, j, 2:4].squeeze(1)
    
    # heading is the tangent of the lane
    heading = torch.atan2(end[..., 1] - start[..., 1], end[..., 0] - start[..., 0])
    
    # position is the center of the lane
    position = ((start + end) / 2)

    if len(local_vec.shape) == 3:
        # boardcast to (M, N, D)
        heading = heading[:, None]
        position = position[:, None]

    # transform local points to sym coord
    local_vec[..., :2] -= position
    local_vec[..., :2] = rotate(local_vec[..., 0], local_vec[..., 1], -heading)

    local_vec[..., 2:4] -= position
    local_vec[..., 2:4] = rotate(local_vec[..., 2], local_vec[..., 3], -heading)

    return local_vec, position, heading


def get_center_vec_init_map(batch, cfg):
    B = len(batch.scene_ids)
    map_cfg = cfg.MAP

    map_inputs, map_masks = [], []
    map_pos, map_head = [], []
    
    for bidx in range(B):
        if type(batch) == SceneBatch:
            local_pos = batch.agent_hist[bidx, 0, -1].position
        else:
            local_pos = batch.agent_hist[bidx, -1].position

        full_vec = batch.extras['vector_lane'][bidx]
        local_vec, local_mask = get_local_vec_map(full_vec, local_pos, map_cfg)

        local_vec, local_pos, local_head = local_map_to_sym_coord(local_vec)

        map_inputs.append(local_vec)
        map_masks.append(local_mask)
        map_pos.append(local_pos)
        map_head.append(local_head)

    all_map = torch.stack(map_inputs)
    all_mask = torch.stack(map_masks)
    all_position = torch.stack(map_pos)
    all_heading = torch.stack(map_head)

    if map_cfg.WITH_TYPE_EMB:
        all_types = all_map[..., 4]
        type_one_hot = torch.zeros_like(all_map[..., :3])
        for type_id in [1, 2, 3]:
            type_one_hot[..., type_id-1] = (all_types == type_id)
        all_map = torch.cat([all_map, type_one_hot], dim=-1)

    if map_cfg.WITH_DIR:
        start = all_map[..., :2]
        end = all_map[..., 2:4]
        diff = end - start
        dir = diff / torch.clip(torch.norm(diff, dim=-1, keepdim=True), min=1e-6)
        all_map = torch.cat([all_map, dir], dim=-1)

    return InputMaskData(all_map, all_mask, all_position, all_heading)

init_map_funcs = {'center_vec': get_center_vec_init_map}

# -----------------------------------------------------------------------------
# init observation formating functions
# -----------------------------------------------------------------------------
# output: produce initial history observation for each scene
# shape: [B, N, D]

def get_agent_temp_obs(agent_data, bidx, nidx, start_t, end_t):
    # default_obs_format = default_trajdata_cfg['obs_format']
    default_obs_format = agent_data['format']


    if start_t < 0:
        his_obs = agent_data['hist'][bidx, nidx, start_t:]
        if end_t > 0:
            fut_obs = agent_data['fut'][bidx, nidx, :end_t]
            center_obs = StateTensor.from_array(torch.cat([his_obs, fut_obs], dim=0), format=default_obs_format)
        else:
            center_obs = his_obs
    else:
        center_obs = agent_data['fut'][bidx, nidx, start_t:end_t]
    
    return center_obs

def get_all_agent_traj(batch: AgentBatch, mode):
    # default_obs_format = default_trajdata_cfg['obs_format']
    default_obs_format = batch.agent_hist._format
    ego_traj = batch.agent_hist[:, None] if mode == 'hist' else batch.agent_fut[:, None]
    neigh_traj = batch.neigh_hist if mode == 'hist' else batch.neigh_fut
    B, N, neigh_T, D = neigh_traj.shape

    if N > 0:
        ego_T = ego_traj.shape[2]
        if neigh_T < ego_T:
            pad_size = ego_T - neigh_T
            pad_hist = torch.full([B, N, pad_size, D], torch.nan)
            if mode == 'hist':
                neigh_traj = torch.cat([pad_hist, neigh_traj], dim=2)
            else:
                neigh_traj = torch.cat([neigh_traj, pad_hist], dim=2)
        return StateTensor.from_array(torch.cat([ego_traj, neigh_traj], dim=1), format=default_obs_format)
    else:
        return ego_traj

def get_all_agent_data(batch):
    agent_data = {}
    if type(batch) == SceneBatch:
        agent_data['names'] = batch.agent_names
        agent_data['hist'] = batch.agent_hist
        agent_data['fut'] = batch.agent_fut
        agent_data['fut_len'] = batch.agent_fut_len
    
        all_extend = torch.cat((batch.agent_hist_extent[..., :2], batch.agent_fut_extent[..., :2]), dim=-2)
        # find the first non-nan value in the extend
        all_extend[all_extend.isnan()] = -1
        all_extend = all_extend.max(dim=-2)[0]
        agent_data['extend'] = all_extend.unsqueeze(-2)
    
    else:
        agent_data['names'] = copy.deepcopy(batch.neigh_names)
        for idx, ego_name in enumerate(batch.agent_name):
            agent_data['names'][idx].insert(0, ego_name)
        agent_data['hist'] = get_all_agent_traj(batch, 'hist')
        agent_data['fut'] = get_all_agent_traj(batch, 'fut')
        agent_data['fut_len'] = torch.cat((batch.agent_fut_len[:, None], batch.neigh_fut_len), dim=1)
        if batch.neigh_hist_extents.shape[1] > 0:
            agent_data['extend'] = torch.cat((batch.agent_hist_extent[:, None, -1:, :2], batch.neigh_hist_extents[:, :, -1:, :2]), dim=1)
        else:
            agent_data['extend'] = batch.agent_hist_extent[:, None, -1:, :2]
    
    agent_data['format'] = agent_data['hist']._format

    return agent_data

def get_batch_temp_obs(agent_data, start_t, end_t):
    # default_obs_format = default_trajdata_cfg['obs_format']
    default_obs_format = agent_data['format']

    if start_t < 0:
        his_obs = agent_data['hist'][:, :, start_t:]
        if end_t > 0:
            fut_obs = agent_data['fut'][:, :, :end_t]
            center_obs = StateTensor.from_array(torch.cat([his_obs, fut_obs], dim=2), format=default_obs_format)
        else:
            center_obs = his_obs
    else:
        center_obs = agent_data['fut'][:, :, start_t:end_t]
    
    return center_obs


def get_center_obs(batch, cfg, start_t, end_t):
    hist_cfg = cfg.HISTORY
    agent_data = get_all_agent_data(batch)

    agent_names = agent_data['names']
    B = len(agent_names)
    # N = max([len(a_name) for a_name in agent_names])

    hist_dim = len(hist_cfg.ELEMENTS.split(','))

    bidxs = []
    oidxs = []
    nidxs = []
    b_aid = []

    all_hist = get_batch_temp_obs(agent_data, start_t, end_t)
    
    if all_hist.shape[2] > 0:
        all_origin = all_hist[:, :, -1]
        nan_origin = all_origin.as_format(hist_cfg.ELEMENTS).isnan().any(dim=-1)
    else:
        all_origin = None
        nan_origin = torch.zeros([B, all_hist.shape[1]], dtype=torch.bool)
    
    for bidx in range(B):
        nidx = 0
        b_aid.append([])
        for oidx in range(all_hist.shape[1]):
            is_target = oidx in batch.tgt_agent_idxs[bidx]

            if (not is_target) and nan_origin[bidx, oidx]:
                continue

            bidxs.append(bidx)
            oidxs.append(oidx)
            nidxs.append(nidx)
            
            b_aid[bidx].append(agent_data['names'][bidx][oidx])
            nidx += 1

    N = max(nidxs) + 1
    b_hist = torch.zeros([B, N, hist_cfg.STEPS, hist_dim], dtype=torch.float) * torch.nan
    b_pos = torch.zeros([B, N, 2], dtype=torch.float)
    b_head = torch.zeros([B, N], dtype=torch.float)

    if all_origin is not None:
        abs_hist = all_hist[bidxs, oidxs]
        abs_origin = all_origin[bidxs, oidxs]
        abs_origin_T = StateTensor.from_array(abs_origin[:, None], format=agent_data['format'])
        rel_hist = transform_to_frame_offset_rot(abs_hist.numpy(), abs_origin_T.numpy())
        rel_hist = StateTensor.from_numpy(rel_hist).as_format(hist_cfg.ELEMENTS).float()
        
        b_pos[bidxs, nidxs] = abs_origin.position
        b_head[bidxs, nidxs] = abs_origin.heading[..., 0]
        b_hist[bidxs, nidxs] = rel_hist
        batch_hist_mask = ~b_hist.isnan()
    else:
        batch_hist_mask = torch.zeros([B, N, hist_cfg.STEPS, hist_dim], dtype=torch.bool)

    if cfg.HISTORY.WITH_EXTEND:
        extent = agent_data['extend'][bidxs, oidxs]
        extent = extent.repeat_interleave(hist_cfg.STEPS, dim=1)
        
        b_extent = torch.zeros([B, N, hist_cfg.STEPS, 2], dtype=torch.float) * torch.nan
        b_extent[bidxs, nidxs] = extent

        b_hist = torch.cat([b_hist, b_extent], dim=-1)
        batch_hist_mask = torch.cat([batch_hist_mask, ~b_extent.isnan()], dim=-1)
    
    if cfg.HISTORY.WITH_AGENT_TYPE:
        agent_type = batch.agent_type[bidxs, oidxs]
        type_one_hot = torch.zeros([len(oidxs), 3], dtype=torch.float)
        for type_id in [1,2,3]:
            type_one_hot[agent_type==type_id, type_id-1] = 1
        type_one_hot = type_one_hot[:, None].repeat_interleave(hist_cfg.STEPS, dim=1)
        
        b_type_one_host = torch.zeros([B, N, hist_cfg.STEPS, 3], dtype=torch.float) * torch.nan
        b_type_one_host[bidxs, nidxs] = type_one_hot

        b_hist = torch.cat([b_hist, b_type_one_host], dim=-1)
        batch_hist_mask = torch.cat([batch_hist_mask, ~b_type_one_host.isnan()], dim=-1)
    
    if cfg.HISTORY.WITH_TIME_EMB:
        T = hist_cfg.STEPS
        b_time_embedding = torch.zeros((B, N, T, T))
        b_time_embedding[:, :, torch.arange(T), torch.arange(T)] = 1

        b_hist = torch.cat([b_hist, b_time_embedding], dim=-1)
        batch_hist_mask = torch.cat([batch_hist_mask, torch.ones((B, N, T, T), dtype=torch.bool)], dim=-1)

    return InputMaskData(b_hist, batch_hist_mask, b_pos, b_head, b_aid)


def get_center_obs_init(batch, cfg):
    return get_center_obs(batch, cfg, -cfg.HISTORY.STEPS, 0)

init_obs_funcs = {'center_history': get_center_obs_init}

# -----------------------------------------------------------------------------
# input/output pair formating functions
# -----------------------------------------------------------------------------
# output: produce input/output pairs for each agent (optional: over T time steps)
# key - batch-agent_id-T
# value - {map: [L, D], obs: [N, D]}


# transform neigh_tgt to relative trajecotry

def abs_traj_to_rel_traj(abs_traj, start_heading, position):
    rel_traj = abs_traj - position
    rel_traj = rotate(rel_traj[..., 0], rel_traj[..., 1], -start_heading)
    return rel_traj

def transform_map_coords_to_local(map_input, local_state):
    local_map = map_input.clone()
    rot_mat = rotation_matrix(-local_state.heading[..., 0])
    local_map[..., :2] = torch.tensor(transform_coords_2d_np_offset_rot(local_map[..., :2], offset=-local_state.position.numpy(), rot_mat=rot_mat))
    local_map[..., 2:4] = torch.tensor(transform_coords_2d_np_offset_rot(local_map[..., 2:4], offset=-local_state.position.numpy(), rot_mat=rot_mat))

    return local_map.float()

def transform_map_obs_tgt_to_local(local_state, abs_tgt, neigh_traj, ego_traj, cfg):
    result = {}
    
    # transform tgt to local coordinates
    rel_traj = transform_to_frame_offset_rot(abs_tgt.numpy(), local_state.numpy())
    result['tgt'] = StateTensor.from_numpy(rel_traj).as_format(cfg.TARGET.ELEMENTS).as_tensor().float()

    # transform neighbor future to local coordinates
    neigh_traj = transform_to_frame_offset_rot(neigh_traj.numpy(), local_state.numpy())
    result['neigh_traj'] = StateTensor.from_numpy(neigh_traj).as_format(cfg.TARGET.ELEMENTS).as_tensor().float()

    # transform ego future to local coordinates
    ego_traj = transform_to_frame_offset_rot(ego_traj.numpy(), local_state.numpy())
    result['ego_traj'] = StateTensor.from_numpy(ego_traj).as_format(cfg.TARGET.ELEMENTS).as_tensor().float()

    result['heading'] = local_state.as_format('h').as_tensor().float()
    result['position'] = local_state.as_format('x,y').as_tensor().float()

    return result

def get_local_io_pairs_T_step_batch(batch, cfg, split):
    #   default_obs_format = default_trajdata_cfg['obs_format']
    default_obs_format = batch.agent_hist._format

    pred_step = cfg.TARGET.STEPS
    sample_rate = cfg.TARGET.SAMPLE_RATE
    tail_padding = cfg.TARGET.TAIL_PADDING

    if tail_padding:
    # padding at the end of the scenario
    # last obs: (s_{T-h}, ..., s_{T-1})
    # last target: (s_{T}, nan, nan, ..., nan)
        max_fut_idx = batch.agent_fut.shape[2] - 1
    else:
    # no padding at the end of the scenario
    # last obs: (s_{T-h-f}, ..., s_{T-f-1})
    # last target: (s_{T-f}, ..., s{T})
        max_fut_idx = batch.agent_fut.shape[2] - pred_step

    fut_indices = np.arange(max_fut_idx + 1)[::sample_rate]

    B = len(batch.agent_names)
    T = len(fut_indices)
    N = max([len(tgt_idxs) for tgt_idxs in batch.tgt_agent_idxs])

    tgt_dim = len(cfg.TARGET.ELEMENTS.split(','))
    goal_dim = len(cfg.GOAL.ELEMENTS.split(','))

    # (TODO): add ego_traj, neigh_traj, goal_map_idx, ego_ext, neigh_ext for collistion loss
    io_pair_batch = {}
    io_pair_batch['heading'] = torch.zeros([B, T, N, 1], dtype=torch.float)
    io_pair_batch['position'] = torch.zeros([B, T, N, 2], dtype=torch.float)
    io_pair_batch['mask'] = torch.zeros([B, T, N], dtype=torch.bool)
    io_pair_batch['tgt'] = torch.zeros([B, T, N, pred_step, tgt_dim], dtype=torch.float) * torch.nan
    io_pair_batch['goal'] = torch.zeros([B, T, N, goal_dim], dtype=torch.float) * torch.nan
    io_pair_batch['agent_type'] = torch.zeros([B, T, N], dtype=torch.long)
    io_pair_batch['init_vel'] = torch.zeros([B, T, N, 2], dtype=torch.float) * torch.nan
    io_pair_batch['extend'] = torch.zeros([B, T, N, 2], dtype=torch.float) * torch.nan
    
    io_pair_batch['agent_names'] = []
    for bidx, b_agent_names in enumerate(batch.agent_names):
        io_pair_batch['agent_names'].append([b_agent_names[oidx] for oidx in batch.tgt_agent_idxs[bidx]])
    io_pair_batch['T_indices'] = fut_indices.tolist()

    center_state_T = []
    center_tgt_T = []
    for t in fut_indices:
        if t == 0:
            center_state = batch.agent_hist[:, :, -1]
        else:
            center_state = batch.agent_fut[:, :, t-1]
        center_state_T.append(center_state)

        center_tgt = batch.agent_fut[:, :, t:t + pred_step]
        # padding nan values to target
        if center_tgt.shape[2] < pred_step:
            pad_size = pred_step - center_tgt.shape[2]
            pad_shape = list(center_tgt.shape)
            pad_shape[2] = pad_size
            center_tgt = torch.cat([center_tgt, torch.full(pad_shape, torch.nan)], dim=2)
        center_tgt_T.append(center_tgt)

    # B, T, O, D
    # O: max number of agents in the batch (O >= N)
    center_state_T = pad_sequence(center_state_T, batch_first=True, padding_value=torch.nan)
    center_tgt_T = pad_sequence(center_tgt_T, batch_first=True, padding_value=torch.nan)
    center_state_T = center_state_T.swapaxes(0, 1)
    center_tgt_T = center_tgt_T.swapaxes(0, 1)
    
    center_state_T = StateTensor.from_array(center_state_T, format=default_obs_format)
    center_tgt_T = StateTensor.from_array(center_tgt_T, format=default_obs_format)

    bidxs = []
    tidxs = []
    oidxs = []
    nidxs = []

    bidxs, tidxs, oidxs, nidxs = zip(*[(bidx, tidx, oidx, nidx)
                                   for bidx in range(B)
                                   for tidx in range(T)
                                   for nidx, oidx in enumerate(batch.tgt_agent_idxs[bidx])])
                
    
    bidxs = torch.tensor(bidxs)
    tidxs = torch.tensor(tidxs)
    oidxs = torch.tensor(oidxs)
    nidxs = torch.tensor(nidxs)

    # filter valid policy agents
    
    # remove agents with nan current state
    state_mask = ~(center_state_T[bidxs, tidxs, oidxs].as_format(cfg.HISTORY.ELEMENTS).isnan().any(dim=-1))
    # remove agents with all-nan target state
    tgt_mask = ~(center_tgt_T[bidxs, tidxs, oidxs].isnan().all(dim=-1).all(dim=-1))

    bidxs = bidxs[state_mask & tgt_mask]
    tidxs = tidxs[state_mask & tgt_mask]
    oidxs = oidxs[state_mask & tgt_mask]
    nidxs = nidxs[state_mask & tgt_mask]

    abs_tgt = center_tgt_T[bidxs, tidxs, oidxs]
    local_state = center_state_T[bidxs, tidxs, oidxs]
    local_state_T = StateTensor.from_array(local_state[:, None], format=default_obs_format)
    rel_tgt = transform_to_frame_offset_rot(abs_tgt.numpy(), local_state_T.numpy())
    rel_tgt = StateTensor.from_numpy(rel_tgt).as_format(cfg.TARGET.ELEMENTS).as_tensor().float()

    fur_len = batch.agent_fut_len[bidxs, oidxs]
    goal = batch.agent_fut[bidxs, oidxs, fur_len-1]
    if cfg.GOAL.LOCAL:
        goal = transform_to_frame_offset_rot(goal.numpy(), local_state.numpy())
        goal = StateTensor.from_numpy(goal)
    
    io_pair_batch['tgt'][bidxs, tidxs, nidxs] = rel_tgt
    io_pair_batch['goal'][bidxs, tidxs, nidxs] = goal.as_format(cfg.GOAL.ELEMENTS).as_tensor().float()
    io_pair_batch['heading'][bidxs, tidxs, nidxs] = local_state.as_format('h').as_tensor().float()
    io_pair_batch['position'][bidxs, tidxs, nidxs] = local_state.as_format('x,y').as_tensor().float()
    io_pair_batch['mask'][bidxs, tidxs, nidxs] = True

    batch_agent_type = batch.agent_type
    batch_agent_type = batch_agent_type[:, None].repeat(1, T, 1)
    io_pair_batch['agent_type'][bidxs, tidxs, nidxs] = batch_agent_type[bidxs, tidxs, oidxs]

    batch_agent_extend = batch.agent_hist_extent[:, :, -1, :2]
    batch_agent_extend = batch_agent_extend[:, None, ...].repeat(1, T, 1, 1)
    io_pair_batch['extend'][bidxs, tidxs, nidxs] = batch_agent_extend[bidxs, tidxs, oidxs]

    init_vel = transform_to_frame_offset_rot(local_state.numpy(), local_state.numpy()).as_format('xd,yd')
    init_vel = StateTensor.from_numpy(init_vel).as_tensor().float()
    io_pair_batch['init_vel'][bidxs, tidxs, nidxs] = init_vel


    io_pair_batch['full_traj_xy'] = torch.zeros([B, N, pred_step*T, 2], dtype=torch.float) * torch.nan
    bidxs_full, oidxs_full, nidxs_full = zip(*[(bidx, oidx, nidx)
                                for bidx in range(B)
                                for nidx, oidx in enumerate(batch.tgt_agent_idxs[bidx])])
    full_traj = batch.agent_fut[bidxs_full, oidxs_full][:, :pred_step*T]
    full_traj = transform_to_frame_offset_rot(full_traj.numpy(), batch.agent_hist[bidxs_full, oidxs_full, -1:].numpy())
    full_traj = StateTensor.from_numpy(full_traj)
    io_pair_batch['full_traj_xy'][bidxs_full, nidxs_full] = full_traj.as_format('x,y').as_tensor().float()
    
    return BatchDataDict(io_pair_batch)

io_pair_funcs = {'local_T_step_batch': get_local_io_pairs_T_step_batch}

# -----------------------------------------------------------------------------
# prompt formating functions
# -----------------------------------------------------------------------------
# output: produce prompt for each agent in each task
# key - task
# value - {prompt: [B, N, D], prompt_mask: [B, N], agent_names: List[List[str]]]}


def get_batch_prompt(scene_batch, cfg):
    all_prompts = dict()
    task = cfg.TASK.TYPES[0]
    # for task in cfg.TASK.TYPES:
    prompt_name = cfg.TASK[task.upper()].PROMPT
    prompt_func = AgentStatusGenerator(cfg.PROMPT[prompt_name.upper()])
    all_prompts[task] = prompt_func.prompt_for_batch(scene_batch)

    return BatchPrompt(all_prompts)

# train/eval data formating functions

# -----------------------------------------------------------------------------
# future observation formating functions
# -----------------------------------------------------------------------------
# output: produce ground-truth observations in future timestamps

def get_future_obs(batch, cfg):
    hist_step = cfg.HISTORY.STEPS
    all_t_indices = batch.extras['all_t_indices']

    fut_type = cfg.FUTURE_OBS_TYPE

    fut_obs = {}
    for idx, end_t in enumerate(all_t_indices):
        if end_t == 0:
            continue
        
        if fut_type == 'latest':
            start_t = end_t - hist_step
        elif fut_type == 'full':
            start_t = -hist_step
        elif fut_type == 'diff':
            start_t = all_t_indices[idx-1]
        
        fut_obs[int(end_t)] = get_center_obs(batch, cfg, start_t, end_t)
    
    return BatchDataDict(fut_obs)

class ImitationBatchFormat(BatchAugmentation):
    def __init__(self, cfg, split):
        self.split = split
        self.cfg = cfg
        f_types = cfg.DATASET.FORMAT.TYPES
        self.init_map_func = init_map_funcs[f_types.INIT_MAP]
        self.init_obs_func = init_obs_funcs[f_types.INIT_OBS]

        self.cond_generator = ConditionGenerator(self.cfg.PROMPT.CONDITION, split)

    def _get_valid_t_indices(self, batch):
        sample_rate = self.cfg.DATASET.FORMAT.TARGET.SAMPLE_RATE
        
        if self.split.upper() == 'ROLLOUT':
            max_time_step = self.cfg.ROLLOUT.POLICY.MAX_STEPS
        else:
            max_time_step = batch.agent_fut.shape[2]

        if self.cfg.DATASET.FORMAT.TARGET.TAIL_PADDING:
            valid_fut_max_idx = max_time_step - 1
        else:
            valid_fut_max_idx = max_time_step - self.cfg.DATASET.FORMAT.TARGET.STEPS
        valid_fut_indices = np.arange(valid_fut_max_idx + 1)[::sample_rate]

        return valid_fut_indices
    
    def _batch_sample_agent_idxs(self, batch_tgt_agent_idxs, max_total_agents, min_scene_agents, split):
        origin_total_agents = np.sum([len(idxs) for idxs in batch_tgt_agent_idxs])
        if origin_total_agents <= max_total_agents:
            return batch_tgt_agent_idxs

        # Select one index for each element
        selected_idxs = {}
        for i, idxs in enumerate(batch_tgt_agent_idxs):
            sample_num = min(len(idxs), min_scene_agents)
            if split.upper() == 'TRAIN' and self.cfg.DATASET.AGENT.RANDOM_TRAIN_SAMPLE:
                selected_idxs[i] = random.sample(idxs, sample_num)
            elif split.upper() == 'VAL' and self.cfg.DATASET.AGENT.RANDOM_VAL_SAMPLE:
                selected_idxs[i] = random.sample(idxs, sample_num)
            else:
                selected_idxs[i] = idxs[:sample_num]
        selected_num = sum([len(idxs) for idxs in selected_idxs.values()])
        remaining_slots = max_total_agents - selected_num

        # Add more indexes randomly without exceeding the limit
        for i, idxs in enumerate(batch_tgt_agent_idxs):
            if remaining_slots <= 0:
                break
            available_idxs = [idx for idx in idxs if idx not in selected_idxs[i]]
            if len(available_idxs) == 0:
                continue
            extra_slots = min(len(available_idxs), remaining_slots)

            if split.upper() == 'TRAIN' and self.cfg.DATASET.AGENT.RANDOM_TRAIN_SAMPLE:
                new_idxs = random.sample(available_idxs, extra_slots)
            elif split.upper() == 'VAL' and self.cfg.DATASET.AGENT.RANDOM_VAL_SAMPLE:
                new_idxs = random.sample(available_idxs, extra_slots)
            else:
                new_idxs = available_idxs[:extra_slots]

            selected_idxs[i] = list(set(selected_idxs[i]) | set(new_idxs))
            remaining_slots -= extra_slots
        
        num_agents = [len(idxs) for idxs in selected_idxs.values()]
        result_total_agents = sum(num_agents)
        print(f'selected agent num: {result_total_agents}/{origin_total_agents}')
        print(f'max scene agent num: {max(num_agents)}')
        print(f'min scene agent num: {min(num_agents)}')

        return list(selected_idxs.values())

    def _filter_tgt_agents(self, batch, split):
        use_sample = self.cfg.DATASET.AGENT.USE_SAMPLE[split.upper()]
        sample_mode = self.cfg.DATASET.AGENT.SAMPLE_MODE
        ego_only = self.cfg.DATASET.AGENT.EGO_ONLY[split.upper()]
        
        for bidx in range(len(batch.tgt_agent_idxs)):
            tgt_agent_idxs = [idx for idx in batch.tgt_agent_idxs[bidx] if batch.agent_fut_len[bidx, idx] > 0]

            # rank the agents by their future length
            batch.tgt_agent_idxs[bidx] = sorted(tgt_agent_idxs, key=lambda x: batch.agent_fut_len[bidx, x], reverse=True)

            if ego_only:
                ego_idx = batch.agent_names[bidx].index('ego')
                batch.tgt_agent_idxs[bidx] = [ego_idx]
        
        if use_sample:
            if sample_mode == 'scene':
                SCENE_MAX_AGENT = self.cfg.DATASET.AGENT.SCENE_MAX_AGENT
                for bidx in range(len(batch.tgt_agent_idxs)):
                    if len(batch.tgt_agent_idxs[bidx]) > SCENE_MAX_AGENT:
                        if self.split.upper() == 'TRAIN' and self.cfg.DATASET.AGENT.RANDOM_TRAIN_SAMPLE:
                            batch.tgt_agent_idxs[bidx] = random.sample(batch.tgt_agent_idxs[bidx], SCENE_MAX_AGENT)
                        else:
                            batch.tgt_agent_idxs[bidx] = batch.tgt_agent_idxs[bidx][:SCENE_MAX_AGENT]
            
            elif sample_mode == 'batch':
                BATCH_MAX_AGENT = self.cfg.DATASET.AGENT.BATCH_MAX_AGENT
                SCENE_MIN_AGENT = self.cfg.DATASET.AGENT.SCENE_MIN_AGENT_SAMPLE
                batch.tgt_agent_idxs = self._batch_sample_agent_idxs(batch.tgt_agent_idxs, BATCH_MAX_AGENT, SCENE_MIN_AGENT, split)
            
            else:
                raise ValueError('Invalid sample mode: {}'.format(sample_mode))

    def apply_agent(self, batch):
        if 'vector_lane' in batch.extras:
            batch.extras['init_map'] = self.init_map_func(batch, self.cfg.DATASET.FORMAT)
        batch.extras['init_obs'] = self.init_obs_func(batch, self.cfg.DATASET.FORMAT)

    def apply_scene(self, batch):
        if self.split.upper() != 'ROLLOUT':
            self._filter_tgt_agents(batch, self.split)
        batch.extras['init_map'] = self.init_map_func(batch, self.cfg.DATASET.FORMAT)
        batch.extras['init_obs'] = self.init_obs_func(batch, self.cfg.DATASET.FORMAT)
        batch.extras['prompt'] = get_batch_prompt(batch, self.cfg)


        if self.split.upper() != 'ROLLOUT':
            batch.extras['io_pairs_batch'] = io_pair_funcs['local_T_step_batch'](batch, self.cfg.DATASET.FORMAT, self.split)
        
        if len(self.cond_generator.cond_funcs) > 0 and self.split.upper() != 'ROLLOUT':
            batch.extras['condition'] = self.cond_generator.get_batch_condition(batch, split=self.split)
        else:
            batch.extras['condition'] = BatchCondition({})

        batch.extras['all_t_indices'] = torch.tensor(self._get_valid_t_indices(batch))
        batch.extras['fut_obs'] = get_future_obs(batch, self.cfg.DATASET.FORMAT)
