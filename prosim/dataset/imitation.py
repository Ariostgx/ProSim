import os
import numpy as np
from tqdm import tqdm
from typing import Union
from functools import partial

from prosim.core.registry import registry
from prosim.dataset.basic import ProSimDataset
from prosim.dataset.data_utils import get_vectorized_lanes
from prosim.dataset.format_utils import ImitationBatchFormat

@registry.register_dataset(name='prosim_imitation')
class ProSimImitationDataset(ProSimDataset):
  def __init__(self, config, split, **args):
    super().__init__(config, split, **args)
  
  def _get_trajdata_cfg(self, cfg, split):
    td_cfg = super()._get_trajdata_cfg(cfg, split)
    td_cfg['centric'] = 'scene'

    # add batch formating augmentation for imitation learning
    if cfg.DATASET.NO_PROCESSING:
      print('do not add batch formating augmentation for imitation learning')
    else:
      td_cfg['augmentations'] = [ImitationBatchFormat(cfg, split)]

    return td_cfg

  def _get_vec_lane_func(self, data_cfg):
    MAP_RANGE = data_cfg.MAP.RANGE[self.split.upper()]

    vec_lane_func = partial(get_vectorized_lanes, 
                            data_cfg=data_cfg,
                            map_range=MAP_RANGE)

    return vec_lane_func