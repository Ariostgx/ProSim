# Copied from https://github.com/ZikangZhou/QCNet/blob/main/utils/geometry.py#L19

import math
import torch

def angle_between_2d_vectors(

        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))

def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def batch_rotate_2D(xy, theta):
  x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
  y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
  return torch.stack([x1, y1], dim=-1)

def rel_traj_coord_to_last_step(traj):
    """
    Convert an arbitray trajectory to a trajectory relative to the last step.
    Args:
        traj: tensor of shape (B, traj_len, 4)
        x, y, sin, cos
    """
    traj_theta = torch.atan2(traj[..., 2], traj[..., 3])

    origin = traj[..., -1, :]
    
    xy_offset = traj[..., :2] - origin[..., None, :2]

    xy_offset = batch_rotate_2D(xy_offset, -traj_theta[..., -1:])

    theta_offset = wrap_angle(traj_theta - traj_theta[..., -1:])
    sin = torch.sin(theta_offset)
    cos = torch.cos(theta_offset)

    rel_traj = torch.cat([xy_offset, sin[..., None], cos[..., None]], dim=-1)

    return rel_traj

def rel_vel_coord_to_last_step(traj, vel):
    """
    Convert a list of vel to a vel relative to the last step.
    Args:
        traj: tensor of shape (B, traj_len, 4)
        vel: tensor of shape (B, traj_len, 2)
        dx, dy
    """
    traj_theta = torch.atan2(traj[..., 2], traj[..., 3])

    rel_vel = batch_rotate_2D(vel, -traj_theta[..., -1:])

    return rel_vel