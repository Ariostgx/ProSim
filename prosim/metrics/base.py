import os
import pytorch_lightning as pl
import torch
import torch.nn as nn

# from ax import Metric
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchmetrics import Accuracy, MeanMetric, Metric

from prosim.core.registry import registry

from pytorch_lightning.callbacks import Callback
from prosim.rollout.distributed_utils import check_mem_usage, print_system_mem_usage, get_gpu_memory_usage

class metric_callback(Callback):
    def __init__(self, config):
        self.config = config
        super().__init__()

    def _shared_update(self, trainer, pl_module, outputs, batch, mode, dataloader_idx):
        cond_sets = pl_module.eval_dataset_cond_sets
        
        model_output = outputs['model_output']
        for task in pl_module.tasks:
            for metric_name in pl_module.metrics[mode][task].keys():
                if len(cond_sets) == 0:
                    pl_module.metrics[mode][task][metric_name](batch, model_output[task])
                else:
                    cond_set = cond_sets[dataloader_idx]
                    pl_module.metrics[mode][task][metric_name][cond_set](batch, model_output[task])

    def _shared_log(self, trainer, pl_module, mode):
        on_step = False
        on_epoch = True
        sync_dist = trainer.num_devices > 1

        cond_sets = pl_module.eval_dataset_cond_sets

        for task in pl_module.tasks:
            for metric_name in pl_module.metrics[mode][task].keys():
                
                if len(cond_sets) == 0:
                    metric_value = pl_module.metrics[mode][task][metric_name].compute()
                    for subname, subvalue in metric_value.items():
                        pl_module.log('{}/metric-{}-{}-{}'.format(mode, task, metric_name, subname), subvalue.detach().cpu().item(), on_epoch=on_epoch, on_step=on_step, sync_dist=sync_dist)
                else:
                    for cond_set in cond_sets:
                        metric_value = pl_module.metrics[mode][task][metric_name][cond_set].compute()
                        for subname, subvalue in metric_value.items():
                            pl_module.log('{}/metric-{}-{}-{}-{}'.format(mode, task, metric_name, cond_set, subname), subvalue.detach().cpu().item(), on_epoch=on_epoch, on_step=on_step, sync_dist=sync_dist)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self._shared_update(trainer, pl_module, outputs, batch, 'val', dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self._shared_update(trainer, pl_module, outputs, batch, 'test', dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._shared_log(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._shared_log(trainer, pl_module, 'test')


@registry.register_metric(name='debug')
class Debug(Metric):
  def __init__(self, config):
      super().__init__()
      self.config = config

  def update(self, batch, model_output):
      pass

  def compute(self):
      return torch.tensor(0)

  def reset(self):
      pass