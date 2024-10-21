import torch

import tqdm
from logging import log
import os
import wandb
import pathlib
import subprocess
from datetime import datetime
import yaml
import yacs.config

import os
import psutil
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from prosim.core.registry import registry

from prosim.rollout import rollout_callback, rollout_callback_gpu, rollout_callback_distributed
from prosim.models.utils import visualization_callback
from prosim.metrics import metric_callback
import multiprocessing
from prosim.dataset.data_utils import _load_road_edge_from_waymo, get_waymo_file_template
from prosim.dataset.condition_utils import ConditionGenerator
import copy


from functools import partial

def get_cond_set_cfg(config, cond_set_name):
    cfg_root_path = os.path.join(config.ROOT_DIR, 'config', 'cond_sampler')
    
    cond_cfg_file = os.path.join(cfg_root_path, cond_set_name + '.yaml')
    
    with open(cond_cfg_file, 'r') as file:
        cond_cfg = yaml.safe_load(file)
    
    new_cond_cfg = dict(config.PROMPT['CONDITION'])
    for key, value in cond_cfg.items():
        new_cond_cfg[key] = value
    
    new_cond_cfg = yacs.config.CfgNode(new_cond_cfg)

    return new_cond_cfg

def check_mem_usage(pid):
  # PID is the process ID of the process you want to check
  process = psutil.Process(pid)
  # Get memory usage information
  memory_info = process.memory_info()
  data = f"RSS: {memory_info.rss / (1024 * 1024)} MB, Shared: {memory_info.shared / (1024 * 1024)} MB"
  return data

def get_system_mem_usage():
  # Get the memory details
  memory = psutil.virtual_memory()

  # Memory usage percentage
  memory_percent = memory.percent

  return memory_percent

def current_time_str():
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H:%M:%S")

    return date_time

def _cache_road_edge(scene_id, save_root, scene_template):
    road_edge = _load_road_edge_from_waymo(scene_id, scene_template)
    save_path = os.path.join(save_root, scene_id + '.npy')
    if os.path.exists(save_path):
        return
    road_edge = np.array(road_edge, dtype=object)
    np.save(save_path, road_edge)
    print(f'{scene_id} saved to: ', save_path)

def cache_batch_vector_lane_tensor(batch, config):
    cache_root = config.DATASET.DATA_PATHS.VECTOR_LANE_CACHE
    env_name = config.DATASET.SOURCE.TRAIN[0]
    env_cache_root = os.path.join(cache_root, env_name)
    os.makedirs(env_cache_root, exist_ok=True)
    
    B = len(batch.scene_ids)
    for idx in range(B):
        scene_id = batch.scene_ids[idx]
        
        cache_path = os.path.join(env_cache_root, scene_id + '.pt')
        if os.path.exists(cache_path):
            continue
        
        vector_lane_tensor = batch.extras['vector_lane'][idx].cpu()
        torch.save(vector_lane_tensor, cache_path)
        print('saved to: ', cache_path)

class BaseTrainer():
    def __init__(self, config, optional_callbacks=[]):
        self.config = config
        self.profier = 'simple'

        avalible_gpus = [i for i in range(torch.cuda.device_count())]

        # use all avalible GPUs
        if self.config.GPU is None or len(self.config.GPU) > torch.cuda.device_count():
            self.config['GPU'] = avalible_gpus
        
        # enable using multiple dataloaders for val and test
        if len(self.config.PROMPT.CONDITION.TYPES) > 0 and len(self.config.PROMPT.CONDITION.EVAL_COND_SETS) > 0:
            self.eval_dataset_cond_sets = self.config.PROMPT.CONDITION.EVAL_COND_SETS
        else:
            self.eval_dataset_cond_sets = []

        # device config
        device_cfg = {}
        if len(self.config.GPU) > 1:
            device_cfg['strategy'] = 'ddp_find_unused_parameters_true'
            device_cfg['devices'] = self.config.GPU
            device_cfg['accelerator'] = 'gpu'
        elif len(self.config.GPU) == 1:
            device_cfg['devices'] = 'auto'
            device_cfg['accelerator'] = 'gpu'
        else:
            device_cfg['devices'] = 'auto'
            device_cfg['accelerator'] = 'cpu'
        self.device_cfg = device_cfg

        print('GPU: ', self.config.GPU)
        print('device_cfg: ', self.device_cfg)

        if len(self.config.GPU) > 0 and self.config.TRAIN.USE_AMP:
            if self.config.TRAIN.USE_BF16:
                self.precision = 'bf16-mixed'
                print('using mixed precision training (bf16-mixed)')
            else:
                self.precision = '16'
                print('using mixed precision training 16')
        else:
            self.precision = '32'

        seed_everything(self.config.SEED, workers=True)
        self._config_save_dir()
        self._set_checkpoint_monitor()
        callbacks = self._config_callbacks()
        self._config_logger(callbacks)
        self._config_data()
        callbacks = self._config_rollout(callbacks)
        self._set_lightning_model()
        self._config_trainer(callbacks)

    def _set_checkpoint_monitor(self):
        self.checkpoint_monitor = 'train/full_loss'
        self.checkpoint_monitor_mode = 'min'

    def _set_lightning_model(self):
        model_cls = registry.get_model(self.config.MODEL.TYPE)
        if self.config.LOAD_CHECKPOINT_MODEL:
            self.lightning_model = model_cls.load_from_checkpoint(
                self.config.LOAD_CHECKPOINT_PATH, config=self.config, strict=False)
            print('load full model from checkpoint: ',
                  self.config.LOAD_CHECKPOINT_PATH)
        else:
            self.lightning_model = model_cls(self.config)

    def _config_data(self):
        dataset_configs = {'train': self.config.TRAIN,
                           'val': self.config.VAL, 'test': self.config.TEST}
        dataset_type = self.config.DATASET.TYPE

        self.data_loaders = {}
        for mode, config in dataset_configs.items():
            dataset = registry.get_dataset(dataset_type)(self.config, config.SPLIT)

            print('GPU in use: ', self.config.GPU)

            batch_size = config.BATCH_SIZE
            
            if len(self.config.GPU) > 1:
                batch_size = int(batch_size / len(self.config.GPU))
            
            print('effective batch size: ', config.BATCH_SIZE)
            print('batch size per GPU: ', batch_size)

            if len(self.eval_dataset_cond_sets) == 0 or mode == 'train':
                # (TODO): having issue with pin_memory on slurm. check later what is going on.
                if self.config.DEBUG:
                    self.data_loaders[mode] = DataLoader(dataset, batch_size=batch_size, shuffle=config.SHUFFLE, pin_memory=False, drop_last=config.DROP_LAST, num_workers=config.NUM_WORKERS, collate_fn=dataset.get_collate_fn())
                    print(f'{mode}: num_workers: ', config.NUM_WORKERS)

                else:
                    self.data_loaders[mode] = DataLoader(dataset, batch_size=batch_size, shuffle=config.SHUFFLE, pin_memory=False, drop_last=config.DROP_LAST, num_workers=config.NUM_WORKERS, collate_fn=dataset.get_collate_fn())
                    print(f'{mode}: num_workers: ', config.NUM_WORKERS)
            
            else:
                self.data_loaders[mode] = []
                for cond_set_name in self.eval_dataset_cond_sets:
                    new_cond_cfg = get_cond_set_cfg(self.config, cond_set_name)
                    cond_dataset = copy.deepcopy(dataset)
                    cond_dataset.augmentations[0].cond_generator = ConditionGenerator(new_cond_cfg, mode)
                    self.data_loaders[mode].append(DataLoader(cond_dataset, batch_size=batch_size, pin_memory=False, drop_last=config.DROP_LAST, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS, collate_fn=cond_dataset.get_collate_fn()))
                
                print(f'{mode} mode evaluate with {len(self.data_loaders[mode])} datasets: ', self.eval_dataset_cond_sets)
        

    def _config_save_dir(self):
        if self.config.EXPERIMENT_NAME:
            self.exp_name = self.config.EXPERIMENT_NAME
        else:
            self.exp_name = current_time_str()

        self.save_dir = os.path.join(
            self.config.SAVE_DIR, self.config.EXPERIMENT_DIR, str(self.exp_name))

        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        print(self.save_dir)

        # save config file
        config_dir = os.path.join(self.save_dir, 'config.yaml')
        with open(config_dir, 'w') as file:
            self.config.dump(stream=file)

    def _config_logger(self, callbacks):
        self.wandb_id = callbacks[-1].details.get("id")


        exp_name = os.path.join(self.config.EXPERIMENT_DIR, self.config.EXPERIMENT_NAME)

        if self.config.DEBUG:
            wandb_proj_name = 'st_debug'
            wandb_id = None
        else:
            wandb_proj_name = self.config.WANDB_PROJ
            wandb_id = self.wandb_id
            print('wandb_id: ', self.wandb_id, flush=True)
            
        self.logger = WandbLogger(
            project=wandb_proj_name, name=exp_name, save_dir=self.save_dir, id=wandb_id)

    def _config_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor, visualization_callback(self.config), metric_callback(self.config)]

        if self.config.SAVE_CHECKPOINT:
            self.checkpoint_callback = ModelCheckpoint(
                monitor=self.checkpoint_monitor,
                mode=self.checkpoint_monitor_mode,
                auto_insert_metric_name=True,
                save_last=True,
                save_top_k=1
            )
            callbacks.append(self.checkpoint_callback)
        
        return callbacks

    def _config_rollout(self, callbacks):
        if self.config.ROLLOUT.ENABLE:
            if self.config.ROLLOUT.MODE == 'gpu':
                callbacks.append(rollout_callback_gpu(self.config, self.rollout_dataset))
            elif self.config.ROLLOUT.MODE == 'distributed':
                callbacks.append(rollout_callback_distributed(self.config, self.rollout_dataset, self.save_dir, self.wandb_id))
            else:
                callbacks.append(rollout_callback(self.config, self.rollout_dataset))
        
        return callbacks


    def _config_trainer(self, callbacks):
        self.trainer = Trainer(max_epochs=self.config.MAX_EPOCHES,
                               log_every_n_steps=self.config.LOG_INTERVAL_STEPS,
                               check_val_every_n_epoch=self.config.VAL_INTERVAL,
                               limit_val_batches=self.config.LIMIT_VAL_BATCHES,
                               default_root_dir=self.save_dir,
                               logger=self.logger,
                               callbacks=callbacks,
                               profiler=self.profier,
                               enable_checkpointing=self.config.SAVE_CHECKPOINT,
                               precision=self.precision,
                               limit_train_batches=self.config.LIMIT_TRAIN_BATCHES,
                               gradient_clip_val=0.5,
                               **self.device_cfg)

    def _save_metric(self, mode='train'):
        save_file = os.path.join(self.save_dir, '{}_metrics.npy'.format(mode))
        try:
            np.save(save_file, self.lightning_model.metric_to_save)
        except:
            print('metric saving failed')

    def _config_test_trainer(self):
        test_trainer = Trainer(log_every_n_steps=self.config.LOG_INTERVAL_STEPS,
                               default_root_dir=self.save_dir,
                               logger=self.logger,
                               profiler=self.profier,
                               precision=self.precision,
                               gradient_clip_val=0.5,
                               **self.device_cfg)

        return test_trainer

    def train(self):
        if self.config.LOAD_CHECKPOINT_TRAINER and self.config.LOAD_CHECKPOINT_PATH is not None:
            ckpt_path = self.config.LOAD_CHECKPOINT_PATH
            print('loading training state from checkpoint: {}'.format(ckpt_path))
        else:
            ckpt_path = None
        self.trainer.fit(self.lightning_model, self.data_loaders['train'], self.data_loaders['val'], ckpt_path=ckpt_path)

    def eval(self):
        if self.config.LOAD_CHECKPOINT_TRAINER and self.config.LOAD_CHECKPOINT_PATH is not None:
            ckpt_path = self.config.LOAD_CHECKPOINT_PATH
            print('loading training state from checkpoint: {}'.format(ckpt_path))
        else:
            ckpt_path = None
        self.trainer.test(model=self.lightning_model, dataloaders=self.data_loaders['test'], ckpt_path=ckpt_path)
        
        self._save_metric('test')

    def data_debug(self):
        import time
        torch.manual_seed(time.time())
        for idx, batch in enumerate(tqdm.tqdm(self.data_loaders['train'], desc='caching road edge polyline')):
            cache_batch_vector_lane_tensor(batch, self.config)