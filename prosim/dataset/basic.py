import os
from tqdm import tqdm
from typing import Union
from functools import partial

from multiprocessing import Pool
import json
import gc
import random
import time
from functools import partial
from pathlib import Path
from typing import (
    List,
    Union,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata.augmentation.augmentation import Augmentation, BatchAugmentation
from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import (
    AgentBatchElement,
    AgentType,
    Scene,
    SceneBatchElement,
    SceneMetadata,
    SceneTime,
    SceneTimeAgent,
)
from trajdata.dataset import UnifiedDataset
from trajdata.dataset_specific import RawDataset
from trajdata.parallel import ParallelDatasetPreprocessor, scene_paths_collate_fn
from trajdata.utils import agent_utils, scene_utils, py_utils

from prosim.dataset.data_utils import get_vectorized_lanes, default_trajdata_cfg, remove_parked, use_all_target, get_waymo_road_edges, get_scene_motion_tag, get_llm_text
from prosim.core.registry import registry
from trajdata.utils import py_utils

import json
from multiprocessing import Pool

def load_motion_tags(file_path):
    with open(file_path, 'r') as f:
        motion_tags = json.load(f)
    return motion_tags
@registry.register_dataset(name='prosim')
class ProSimDataset(UnifiedDataset):
  def __init__(self, config, split, **args):
      self.cfg = config
      self.split = split
      td_cfg = self._get_trajdata_cfg(config, split)

      self.skip_cache_check = config.DATASET.SKIP_CACHE_CHECK

      for name, value in args.items():
        td_cfg[name] = value

      super().__init__(**td_cfg)

      print(self.cfg.DATASET.DATA_LIST.MODE)
      if self.cfg.DATASET.DATA_LIST.MODE == 'scene_ts':
        self._filter_scene_ts()
      
      self._subsample_data(init=True)
  
  def _get_trajdata_cfg(self, cfg, split):
      td_cfg = default_trajdata_cfg.copy()
      data_cfg = cfg.DATASET
      MODE = split.upper()

      if data_cfg.DATA_LIST.MODE != 'all':
        list_root = data_cfg.DATA_LIST.ROOT
        data_list = os.path.join(list_root, data_cfg.DATA_LIST[MODE])
        with open(data_list, 'r') as f:
          self.ids_in_list = [line.strip() for line in f.readlines()]
      
      if data_cfg.DATA_LIST.MODE == 'scene':
        self.select_ids = self.ids_in_list
        self.select_logs = None
        print('{}: select {} scenes from {}'.format(MODE, len(self.select_ids), data_list))
      elif data_cfg.DATA_LIST.MODE == 'log':
        self.select_ids = None
        self.select_logs = self.ids_in_list
        print('{} select {} logs'.format(MODE, len(self.select_logs)))
      elif data_cfg.DATA_LIST.MODE == 'all':
        self.select_ids = None
        self.select_logs = None
        print('{}: use all scenes'.format(MODE))
      elif data_cfg.DATA_LIST.MODE == 'scene_ts':
        scene_ids = [line.split('_')[0] for line in self.ids_in_list]
        self.select_ids = list(set(scene_ids))
        self.select_logs = None
        print('{}: select {} scenes from {}'.format(MODE, len(self.select_ids), data_list))
      else:
        self.select_ids = None
        self.select_logs = None
        print('data list mode: {}'.format(data_cfg.DATA_LIST.MODE))
      
      if data_cfg.USE_PED_CYCLIST:
        td_cfg['only_types'] = [AgentType.VEHICLE, AgentType.PEDESTRIAN, AgentType.BICYCLE]
         
      
      # config sample rate
      self.sample_rate = data_cfg.SCENE.SAMPLE_RATE[MODE]
      
      td_cfg['save_index'] = False if self.cfg.DATASET.DATA_LIST.MODE == 'scene_ts' else True

      td_cfg['desired_data'] = data_cfg.SOURCE[MODE]
      td_cfg['desired_dt'] = data_cfg.MOTION.DT
      td_cfg['history_sec'] = (data_cfg.MOTION.HISTORY_SEC, data_cfg.MOTION.HISTORY_SEC)
      td_cfg['future_sec'] = (data_cfg.MOTION.FUTURE_SEC[MODE], data_cfg.MOTION.FUTURE_SEC[MODE])
      td_cfg['num_workers'] = cfg[MODE].NUM_WORKERS
      td_cfg['data_dirs'] = {source: data_cfg.DATA_PATHS[source.upper().replace('-', '_')] for source in data_cfg.SOURCE[MODE]}
      td_cfg['ego_only'] = data_cfg.USE_EGO_CENTER[MODE]
      td_cfg['use_all_agents'] = data_cfg.USE_ALL_AGENTS
      td_cfg['incl_vector_map'] = data_cfg.LOAD_VEC_MAP[MODE]

      if split.upper() == 'ROLLOUT':
        td_cfg['transforms'] = []
      elif data_cfg.REMOVE_PARKED[MODE]:
        td_cfg['transforms'] = [remove_parked]
      else:
        td_cfg['transforms'] = [use_all_target]

      if 'CACHE_PATH' in data_cfg:
        td_cfg['cache_location'] = data_cfg.CACHE_PATH
        print('cache location: {}'.format(data_cfg.CACHE_PATH))
      
      if data_cfg.CACHE_MAP == False:
        td_cfg['require_map_cache'] = False
        td_cfg['incl_raster_map'] = False
        print('do not cache map')
      else:
        td_cfg['incl_raster_map'] = data_cfg.USE_RASTER_MAP
        if split.upper() == 'TRAIN':
          td_cfg['incl_raster_map'] = False
          print('do not use raster map for training')
      
      if data_cfg.NO_PROCESSING:
        print('do not add extra functions for data processing')
      else:
        td_cfg['extras'] = self._get_extra_funcs(data_cfg)

      return td_cfg


  def _get_extra_funcs(self, data_cfg):
      funcs = {}
      funcs['vector_lane'] = self._get_vec_lane_func(data_cfg)

      if data_cfg.USE_WAYMO_ROAD_EDGE:
        funcs['road_edge_polyline'] = partial(get_waymo_road_edges, 
                             config=data_cfg, split=self.split.upper())
      
      if data_cfg.USE_MOTION_TAGS:
        motion_tag_path = data_cfg.DATA_PATHS.MOTION_TAGS[self.split.upper()]

        if 'waymo' in motion_tag_path:
          all_motion_tags = None

        else:
          with open(motion_tag_path, 'r') as f:
            all_motion_tags = json.load(f)

        funcs['motion_tag'] = partial(get_scene_motion_tag, config=self.cfg, all_motion_tags=all_motion_tags, split=self.split)
      
      cond_cfg = self.cfg.PROMPT.CONDITION
      if ('llm_text' in cond_cfg.TYPES or 'llm_text_OneText' in cond_cfg.TYPES):
        funcs['llm_texts'] = partial(get_llm_text, config=self.cfg, split=self.split)

      return funcs

  def _get_vec_lane_func(self, map_cfg):
      MAP_RANGE = map_cfg.RANGE[self.split.upper()]
      vec_lane_func = partial(get_vectorized_lanes, 
                             SAMPLE_RATE=map_cfg.SAMPLE_RATE,
                             MAP_RANGE=MAP_RANGE,
                             centric_type='agent')
      
      return vec_lane_func

  def _filter_scene_ts(self):
    # {scene_id}_{scene_ts}
    filterd_data_index = []

    for i, (scene_file, scene_ts) in enumerate(self._data_index):
      scene_id = scene_file.split('/')[-2]
      if '{}_{}'.format(scene_id, scene_ts) in self.ids_in_list:
        filterd_data_index.append(self._data_index[i])
    
    self._data_index = filterd_data_index
    self._data_len = len(self._data_index)


  def _subsample_data(self, init=False):
    if self.sample_rate is None:
      print('{}: no sample rate specified, use all {} frames'.format(self.split.upper(), self._data_len))
      return
    
    self.sample_rate = int(self.sample_rate)
    assert self.sample_rate > 0

    if init:
      self.ori_len = self._data_len
      self.ori_data_index = self._data_index
      self.ori_scene_index = self._scene_index

    self._data_index = [self.ori_data_index[i] for i in range(0, self.ori_len, self.sample_rate)]
    self._data_len = len(self._data_index)
    self._scene_index = [self.ori_scene_index[i] for i in range(0, len(self.ori_scene_index), self.sample_rate)]

    print('{}: Uniform Sample with rate: {}, sample {} from {} frames'.format(self.split.upper(), self.sample_rate, self._data_len, self.ori_len))
     
  
  def get_desired_scenes_from_env(
    self,
    scene_tags,
    scene_description_contains,
    env):
    scenes_list = list()
    for scene_tag in tqdm(
        scene_tags, desc=f"Getting Scenes from {env.name}", disable=not self.verbose
    ):
        if env.name in scene_tag:
            scenes_list += env.get_matching_scenes(
                scene_tag,
                scene_description_contains,
                self.env_cache,
                self.rebuild_cache,
            )

    if self.select_ids is not None:
      ori_len = len(scenes_list)
      scenes_list = [s for s in scenes_list if s.name in self.select_ids]
      print('{}: filter scences with {} scene_ids: {}/{}'.format(self.split.upper(), len(self.select_ids), len(scenes_list), ori_len))
    
    if self.select_logs is not None:
      ori_len = len(scenes_list)
      scenes_list = [s for s in scenes_list if s.name.split('=')[0] in self.select_logs]
      print('{}: filter {} logs from scene_logs: {}/{}'.format(self.split.upper(), len(self.select_logs), len(scenes_list), ori_len))

    return scenes_list

  def check_cache_status(self, env, scenes_list):
    if self.skip_cache_check:
      return True, True
    else:
       return super().check_cache_status(env, scenes_list)

  def _index_cache_path(
      self, ret_args: bool = False):
      # Whichever UnifiedDataset arguments affect data indexing are captured
      # and hashed together here.
      impactful_args = {
          "desired_data": tuple(self.desired_data),
          "scene_description_contains": tuple(self.scene_description_contains)
          if self.scene_description_contains is not None
          else None,
          "centric": self.centric,
          "desired_dt": self.desired_dt,
          "history_sec": self.history_sec,
          "future_sec": self.future_sec,
          "incl_robot_future": self.incl_robot_future,
          "only_types": tuple(t.name for t in self.only_types)
          if self.only_types is not None
          else None,
          "only_predict": tuple(t.name for t in self.only_predict)
          if self.only_predict is not None
          else None,
          "no_types": tuple(t.name for t in self.no_types)
          if self.no_types is not None
          else None,
          "ego_only": self.ego_only,
      }

      if self.select_ids is not None:
        impactful_args['select_ids'] = tuple(self.select_ids)
      else:
        impactful_args['select_ids'] = []
      
      if self.select_logs is not None:
        impactful_args['select_logs'] = tuple(self.select_logs)
      else:
        impactful_args['select_logs'] = []

      index_hash: str = py_utils.hash_dict(impactful_args)
      index_cache_path = self.cache_path / "data_indexes" / index_hash

      if ret_args:
          return index_cache_path, impactful_args
      else:
          return index_cache_path

  def _get_scene_path_info(self, idx: int):
    if self.centric == "scene":
      scene_path, ts = self._data_index[idx]
      agent_id = None
    elif self.centric == "agent":
      scene_path, agent_id, ts = self._data_index[idx]
    
    return scene_path, agent_id, ts

  def getitem_helper(self, idx: int) -> Union[SceneBatchElement, AgentBatchElement]:
      if self._cached_batch_elements is not None:
          return self._cached_batch_elements[idx]

      scene_path, agent_id, ts = self._get_scene_path_info(idx)
      
      scene: Scene = EnvCache.load(scene_path)
      scene_utils.enforce_desired_dt(scene, self.desired_dt)
      scene_cache: SceneCache = self.cache_class(
          self.cache_path, scene, self.augmentations
      )
      scene_cache.set_obs_format(self.obs_format)
      if self.centric == "scene":
          scene_time: SceneTime = SceneTime.from_cache(
              scene,
              ts,
              scene_cache,
              only_types=self.only_types,
              no_types=self.no_types,
          )

          batch_element: SceneBatchElement = SceneBatchElement(
              scene_cache,
              idx,
              scene_time,
              self.history_sec,
              self.future_sec,
              self.agent_interaction_distances,
              self.incl_robot_future,
              self.incl_raster_map,
              self.raster_map_params,
              self._map_api,
              self.vector_map_params,
              self.state_format,
              self.standardize_data,
              self.standardize_derivatives,
              self.max_agent_num,
              self.use_all_agents,
              self.only_types,
              self.no_types,
          )
          batch_element.map_name = f"{scene_time.scene.env_name}:{scene_time.scene.location}"
      elif self.centric == "agent":
          scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
              scene,
              ts,
              agent_id,
              scene_cache,
              only_types=self.only_types,
              no_types=self.no_types,
              incl_robot_future=self.incl_robot_future,
          )

          batch_element: AgentBatchElement = AgentBatchElement(
              scene_cache,
              idx,
              scene_time_agent,
              self.history_sec,
              self.future_sec,
              self.agent_interaction_distances,
              self.incl_robot_future,
              self.incl_raster_map,
              self.raster_map_params,
              self._map_api,
              self.vector_map_params,
              self.state_format,
              self.standardize_data,
              self.standardize_derivatives,
              self.max_neighbor_num,
          )

      for key, extra_fn in self.extras.items():
          batch_element.extras[key] = extra_fn(batch_element)
      for transform_fn in self.transforms:
          batch_element = transform_fn(batch_element)
      if not self.vector_map_params.get("collate", False):
          batch_element.vec_map = None

      return batch_element

  def __getitem__(self, idx: int) -> Union[SceneBatchElement, AgentBatchElement]:
    element = self.getitem_helper(idx)

    # skip if there is no tgt agent
    if len(element.tgt_agent_idx) == 0:
      print('skip scene {} because there is no tgt agent'.format(element.scene_id), flush=True)
      next_idx = min(self._data_len-1, idx+1)
      return self.__getitem__(next_idx)

    # skip if there is no map
    if 'vector_lane' in element.extras and element.extras['vector_lane'] is None:
      print('skip scene {} because there is no map'.format(element.scene_id), flush=True)
      next_idx = min(self._data_len-1, idx+1)
      return self.__getitem__(next_idx)

    if 'waymo' in self.envs[0].name and self._map_api is not None:
      del self._map_api.maps
      self._map_api.maps = {}

    return element

  def check_is_cached(self, scene_info):
      """Function to check if the scene is cached."""
      return self.env_cache.scene_is_cached(
          scene_info.env_name,
          scene_info.name,
          self.desired_dt if self.desired_dt is not None else scene_info.dt
      )

  # Function to run the cache checking in parallel
  def check_scenes_in_parallel(self, scenes_list):
      cached_scenes = []

      # Using Pool for parallel processing
      with Pool(64) as pool:
          results = list(tqdm(pool.imap(self.check_is_cached, scenes_list), total=len(scenes_list), desc="Checking cached scenes", disable=not self.verbose))
      
      for scene, is_cached in zip(scenes_list, results):
          if is_cached:
              cached_scenes.append(scene)

      # Determine uncached scenes
      uncached_scenes = [s for s in scenes_list if s not in cached_scenes]
      return cached_scenes, uncached_scenes

  def preprocess_scene_data(
      self,
      scenes_list: Union[List[SceneMetadata], List[Scene]],
      num_workers: int,
  ) -> List[Path]:
      # List of (Original cached path, Temporary cached path)
      scene_paths: List[Path] = list()

      cached_scenes = []
      uncached_scenes = []
      # Check if the scenes are cached
      for scene_info in tqdm(scenes_list, desc='check cached scenes (Serially)', disable=not self.verbose):
        scene_path: Path = EnvCache.scene_metadata_path(
          self.cache_path,
          scene_info.env_name,
          scene_info.name,
          self.desired_dt,
        )
        if self.skip_cache_check or scene_path.exists():
          scene_paths.append(scene_path)
          cached_scenes.append(scene_info)
        else:
          uncached_scenes.append(scene_info)

      print('cached: {}, uncached: {}'.format(len(cached_scenes), len(uncached_scenes)), flush=True)

      all_cached: bool = not self.rebuild_cache and len(uncached_scenes) == 0

      serial_scenes: List[SceneMetadata]
      parallel_scenes: List[SceneMetadata]
      
      # pass the cached scences to serial, pass the uncached scences to parallel
      if num_workers > 1 and not all_cached:
          serial_scenes = []
          parallel_scenes = uncached_scenes

          # Fixing the seed for random suffling (for debugging and reproducibility).
          shuffle_rng = random.Random(123)
          shuffle_rng.shuffle(parallel_scenes)
      else:
          serial_scenes = uncached_scenes
          parallel_scenes = list()

      if serial_scenes:
          # Scenes for which it's faster to process them serially. See
          # the longer comment below for a more thorough explanation.
          scene_info: SceneMetadata
          for scene_info in tqdm(
              serial_scenes,
              desc="Calculating Agent Data (Serially)",
              disable=not self.verbose,
          ):
              scene_dt: float = (
                  self.desired_dt if self.desired_dt is not None else scene_info.dt
              )
              if not self.rebuild_cache and scene_info in cached_scenes:
                # This is a fast path in case we don't need to
                # perform any modifications to the scene_info.
                scene_path: Path = EnvCache.scene_metadata_path(
                    self.cache_path,
                    scene_info.env_name,
                    scene_info.name,
                    scene_dt,
                )

                scene_paths.append(scene_path)
                continue

              corresponding_env: RawDataset = self.envs_dict[scene_info.env_name]
              scene: Scene = agent_utils.get_agent_data(
                  scene_info,
                  corresponding_env,
                  self.env_cache,
                  self.rebuild_cache,
                  self.cache_class,
                  self.desired_dt,
              )

              scene_path: Path = EnvCache.scene_metadata_path(
                  self.cache_path, scene.env_name, scene.name, scene.dt
              )
              scene_paths.append(scene_path)

      # Done with these lists. Cutting memory usage because
      # of multiprocessing below.
      del serial_scenes
      scenes_list.clear()

      # No more need for the original dataset objects and freeing up
      # this memory allows the parallel processing below to run very fast.
      # The dataset objects for any envs used below will be loaded in each
      # process.
      env_dict = {}
      for env in self.envs:
          if 'nuplan' not in env.name:
              env.del_dataset_obj()
          env_dict[env.name] = env
      if parallel_scenes:
          parallel_preprocessor = ParallelDatasetPreprocessor(
              parallel_scenes,
              {
                  env_name: str(env.metadata.data_dir)
                  for env_name, env in self.envs_dict.items()
              },
              str(self.env_cache.path),
              self.desired_dt,
              self.cache_class,
              self.rebuild_cache,
              env_dict
          )

          del parallel_scenes

          gc.collect()

          dataloader = DataLoader(
              parallel_preprocessor,
              batch_size=1,
              num_workers=num_workers,
              shuffle=False,
              collate_fn=scene_paths_collate_fn,
          )

          for processed_scene_paths in tqdm(
              dataloader,
              desc=f"Calculating Agent Data ({num_workers} CPUs)",
              disable=not self.verbose,
          ):
              scene_paths += [
                  Path(path_str)
                  for path_str in processed_scene_paths
                  if path_str is not None
              ]

      return scene_paths