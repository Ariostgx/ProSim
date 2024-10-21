import numpy as np
import pandas as pd

from shapely import affinity
from shapely.geometry import Polygon
from pandas.core.series import Series

from trajdata.simulation.sim_metrics import SimMetric
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
class CrashDetect(SimMetric):
  def __init__(self, tgt_agent_ids: List[str], agent_extends: Dict[str, List[float]], iou_threshold=0.1, mode='sim') -> None:
    super().__init__("crash_detect")
    
    self.tgt_agent_ids = tgt_agent_ids
    self.agent_extends = agent_extends
    self.iou_threshold = iou_threshold
    self.mode = mode

  def _get_box_polygon(self, agent: Series, extents: List[float]):
    box_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    box_points[:, 0] = box_points[:, 0] * extents[1]
    box_points[:, 1] = box_points[:, 1] * extents[0]
    box = Polygon(box_points)

    # Get the agent polygon
    box = affinity.rotate(box, agent['heading'], origin='centroid')
    box = affinity.translate(box, agent['x'], agent['y'])
    return box

  def _poly_iou_check(self, poly1: Polygon, poly2: Polygon, iou_threshold: float):
    if not poly1.intersects(poly2):
      return False

    union = poly1.union(poly2).area
    inter = poly1.intersection(poly2).area
    iou = inter / union

    return iou > iou_threshold

  def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame):
    data = sim_df if self.mode == 'sim' else gt_df

    all_scene_ts = data.index.get_level_values('scene_ts').unique()

    crash_logs = {agent_id: [] for agent_id in self.tgt_agent_ids}

    for scene_ts in all_scene_ts:
      scene_data = data.xs(scene_ts, level='scene_ts')
      frame_agent_ids = scene_data.index.get_level_values('agent_id').unique()

      # create polygons for all agents in this frame
      agent_boxes = {}
      for agent_id in frame_agent_ids:
        agent = scene_data.loc[agent_id]

        # Get the extents for the agent
        agent_extents = self.agent_extends[agent_id]

        # Get the box polygon
        box = self._get_box_polygon(agent, agent_extents)
        agent_boxes[agent_id] = box

      for target_id in self.tgt_agent_ids:
        # target agent does not exist in this frame
        if target_id not in frame_agent_ids:
          crash_logs[target_id].append(0)
          continue

        target_box = agent_boxes[target_id]

        for agent_id in frame_agent_ids:
          if agent_id == target_id:
            continue

          agent_box = agent_boxes[agent_id]

          if self._poly_iou_check(target_box, agent_box, self.iou_threshold):
            crash_logs[target_id].append(1)
            break

        if scene_ts not in crash_logs[target_id]:
          crash_logs[target_id].append(0)
    
    crash_detect = {}
    for agent_id in self.tgt_agent_ids:
      crash_detect[agent_id] = int(crash_logs[agent_id].count(1) > 0)

    return crash_detect

class GoalReach(SimMetric):
  def __init__(self, tgt_agent_ids: List[str], dist_threshold=2.0) -> None:
    super().__init__("goal_reach")
    
    self.tgt_agent_ids = tgt_agent_ids
    self.dist_threshold = dist_threshold
  
  def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame):
    gt_ts = gt_df.index.get_level_values('scene_ts').unique()
    sim_ts = sim_df.index.get_level_values('scene_ts').unique()
    last_ts = min(gt_ts[-1], sim_ts[-1])

    goal_reach = {}

    gt_last_frame = gt_df.xs(last_ts, level='scene_ts')
    gt_agent_ids = gt_last_frame.index.get_level_values('agent_id').unique()
    
    sim_last_frame = sim_df.xs(last_ts, level='scene_ts')
    sim_agent_ids = sim_last_frame.index.get_level_values('agent_id').unique()

    for agent_id in self.tgt_agent_ids:
      if agent_id not in gt_agent_ids or agent_id not in sim_agent_ids:
        continue
      
      gt_agent = gt_last_frame.loc[agent_id]
      sim_agent = sim_last_frame.loc[agent_id]

      gt_pos = np.array([gt_agent['x'], gt_agent['y']])
      sim_pos = np.array([sim_agent['x'], sim_agent['y']])

      dist = np.linalg.norm(gt_pos - sim_pos)

      goal_reach[agent_id] = dist < self.dist_threshold

    return goal_reach