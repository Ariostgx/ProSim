import math
import torch

def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def batch_rotate_2D(xy, theta):
  x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
  y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
  return torch.stack([x1, y1], dim=-1)


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
  # (TODO): this could be slow as Q is large; consider optimizing this
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