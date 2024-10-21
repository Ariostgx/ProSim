import math
from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR, LambdaLR
import numpy as np
import wandb
import pytorch_lightning as pl

from prosim.core.registry import registry
from prosim.loss.loss_func import loss_func_dict

class LinearWarmupCosineAnnealingLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay. After linear warmup,
    the LR decays as:
    .. math::
        \eta_t = \eta_{max}\cos^2(\frac{T_{cur} - T_{warm}}{T_{max} - T_{warm}}\frac{\pi}{2})
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)

class BaseModel(pl.LightningModule):
    def __init__(self, config):
      super().__init__()
      self.config = config
      self.tasks = config.TASK.TYPES
      
      if len(self.config.PROMPT.CONDITION.TYPES) > 0 and len(self.config.PROMPT.CONDITION.EVAL_COND_SETS) > 0:
        self.eval_dataset_cond_sets = self.config.PROMPT.CONDITION.EVAL_COND_SETS
      else:
        self.eval_dataset_cond_sets = []

      self._config_metric()
      self._config_loss_functions()
      self.lr = self.config.TRAIN.LR


    def _config_parameters(self):
      self.model_params = []
      self.lora_params = []
      self.adapter_params = []
      self.goal_pred_params = []
      self.cond_params = []

      is_lora = lambda x: ('lora' in x)
      is_adapter = lambda x: ('prompt_to_llm' in x) or ('llm_to_cond' in x) or ('ln_prompt' in x)
      is_goal_pred = lambda x: ('pred_mlp' in x) or ('goal_prob_head' in x) or ('goal_point_head' in x)
      is_cond_transformer = lambda x: ('condition_encoder' in x) or ('condition_attn' in x)

      if self.lr == 0:
        use_diff_lr = True
      else:
        use_diff_lr = self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.LORA_LR_SCALE != 1.0 or self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.ADAPTER_LR_SCALE != 1.0 or self.config.LOSS.ROLLOUT_TRAJ.GOAL_MODEL_LR_SCALE != 1.0 or self.config.MODEL.CONDITION_TRANSFORMER.LR_SCALE != 1.0

      for model in self.models:
        named_params = [p for p in model.named_parameters() if p[1].requires_grad]

        if use_diff_lr:
            self.lora_params += [param for name, param in named_params if is_lora(name)]
            self.adapter_params += [param for name, param in named_params if is_adapter(name)]
            self.goal_pred_params += [param for name, param in named_params if is_goal_pred(name)]
            self.cond_params += [param for name, param in named_params if is_cond_transformer(name)]
            self.model_params += [param for name, param in named_params if not is_lora(name) and not is_adapter(name) and not is_goal_pred(name) and not is_cond_transformer(name)]
        else:
            self.model_params += [param for name, param in named_params]

      self.params = [{'params': self.model_params, 'lr': self.lr, 'name': 'model'}]

      # allow fixing the main model while training the special parts
      lr_base = self.lr if self.lr > 0 else 1e-3

      if len(self.lora_params) > 0:
        lora_lr = lr_base * self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.LORA_LR_SCALE
        self.params.append({'params': self.lora_params, 'lr': lora_lr, 'name': 'lora'})

        print('LORA lr: {}'.format(lora_lr))
    
      if len(self.adapter_params) > 0:
        adapter_lr = lr_base * self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.ADAPTER_LR_SCALE
        self.params.append({'params': self.adapter_params, 'lr': adapter_lr, 'name': 'adapter'})

        print('Adapter lr: {}'.format(adapter_lr))
    
      if len(self.goal_pred_params) > 0:
        goal_pred_lr = lr_base * self.config.LOSS.ROLLOUT_TRAJ.GOAL_MODEL_LR_SCALE
        self.params.append({'params': self.goal_pred_params, 'lr': goal_pred_lr, 'name': 'goal_pred'})

        print('Goal prediction lr: {}'.format(goal_pred_lr))
    
      if len(self.cond_params) > 0:
        cond_lr = lr_base * self.config.MODEL.CONDITION_TRANSFORMER.LR_SCALE
        self.params.append({'params': self.cond_params, 'lr': cond_lr, 'name': 'condition transformer'})

        print('Condition lr: {}'.format(cond_lr))

    def on_save_checkpoint(self, checkpoint):
        # Modify checkpoint to exclude llm_model parameters but include lora model parameters

        is_trainable = lambda x: ('lora' in x) or ('llm_model' not in x)

        checkpoint['state_dict'] = {k: v for k, v in self.state_dict().items() if is_trainable(k)}
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False) -> None:
        # Load model state dict from checkpoint, set strict to False to ignore missing keys
        for key in list(state_dict.keys()):
           if 'lora' in key:
               print('loading lora weight: {}'.format(key))

        return super().load_state_dict(state_dict, strict)

    def _config_metric(self):
        task_config = self.config
        self.metrics = {}

        for mode in ['train', 'val', 'test']:
            self.metrics[mode] = {}
        
            for task in self.tasks:
                self.metrics[mode][task] = {}
                metric_types = self.config.TASK[task.upper()].METRICS

                for metric_type in metric_types:
                    if len(self.eval_dataset_cond_sets) == 0 or mode == 'train':
                        self.metrics[mode][task][metric_type] = registry.get_metric(metric_type)(task_config)
                        metric_name = '{}-{}-{}'.format(mode, task, metric_type)
                        setattr(self, 'metric-{}'.format(metric_name), self.metrics[mode][task][metric_type])
                    else:
                        self.metrics[mode][task][metric_type] = {}
                        for cond_set in self.eval_dataset_cond_sets:
                            self.metrics[mode][task][metric_type][cond_set] = registry.get_metric(metric_type)(task_config)
                            metric_name = '{}-{}-{}-{}'.format(mode, task, metric_type, cond_set)
                            setattr(self, 'metric-{}'.format(metric_name), self.metrics[mode][task][metric_type][cond_set])

    def _config_loss_functions(self):
       self.loss_funcs = {}
       for task in self.tasks:
        task_loss_name = self.config.TASK[task.upper()].LOSS
        self.loss_funcs[task] = loss_func_dict[task_loss_name]

    def _compute_loss(self, batch, model_output):
      all_loss = {'full_loss': 0}
      
      for task in self.tasks:
        weight = self.config.TASK[task.upper()].WEIGHT
        loss = self.loss_funcs[task](batch, model_output[task], self.config)
        for name, value in loss.items():
            if name == 'full_loss':
                all_loss['full_loss'] += value * weight
            all_loss['{}-{}'.format(task, name)] = value * weight

      return all_loss

    def forward(self, batch):
      raise NotImplementedError

    def _batch_forward(self, batch, mode, batch_idx):
      model_output = self.forward(batch, mode)
      loss = self._compute_loss(batch, model_output)

      if mode == 'train':
          on_step = True
          on_epoch = False
          sync_dist = False
          rank_zero_only = True
      else:
          on_step = False
          on_epoch = True
          sync_dist = True if len(self.config.GPU) > 1 else False
          rank_zero_only = False

      for loss_name, value in loss.items():
        self.log('{}/{}'.format(mode, loss_name), value.detach().cpu().item(), on_step=on_step,
                    on_epoch=on_epoch, sync_dist=sync_dist, batch_size=len(batch.scene_ids), rank_zero_only=rank_zero_only)

      return {'loss': loss['full_loss'], 'model_output': model_output}
    
    def training_step(self, batch, batch_idx):
        return self._batch_forward(batch, 'train', batch_idx)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        
        return self._batch_forward(batch, 'val', batch_idx)
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._batch_forward(batch, 'test', batch_idx)

    def _configure_schedulers(self, optimizer, scheduler_config):
      scheduler_name = scheduler_config.TYPE
      
      if scheduler_name == 'StepLR':
          scheduler = StepLR(
              optimizer, step_size=scheduler_config.STEP, gamma=scheduler_config.GAMMA)
          
          print('using StepLR scheduler')
      
      elif scheduler_name == 'MultiStepLR':
          scheduler = MultiStepLR(
              optimizer, milestones=scheduler_config.MILESTONES, gamma=scheduler_config.GAMMA)
          
          print('using multistep lr scheduler')

      elif scheduler_name == 'PatienceMultiplicativeLR':
          patience_epoch = scheduler_config.STEP

          def patience_gamma(epoch):
              if epoch >= patience_epoch:
                  return scheduler_config.GAMMA ** (epoch - patience_epoch + 1)
              else:
                  return 1.0

          scheduler = LambdaLR(optimizer, lr_lambda=patience_gamma)

          print('using patience scheduler: GAMMA = {}, start_step = {}'.format(scheduler_config.GAMMA, patience_epoch))

      elif scheduler_name == 'CosineAnneling':
          eta_min = scheduler_config.ETA_MIN
          if scheduler_config.T_MAX is not None:
              T_max = scheduler_config.T_MAX
          else:
              T_max = self.config.MAX_EPOCHES

          scheduler = CosineAnnealingLR(
              optimizer, T_max=T_max, eta_min=eta_min)
          
          print('using CosineAnnealingLR: eta_min={}, T_max={}'.format(eta_min, T_max))
    
      elif scheduler_name == 'LinearWarmupCosineAnnealingLR':
          eta_min = scheduler_config.ETA_MIN
          max_steps = scheduler_config.MAX_STEPS
          warmup_steps = scheduler_config.WARMUP_STEPS
    
          scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, max_steps, warmup_steps)
            
          print('using LinearWarmupCosineAnnealingLR: eta_min=0, max_steps={}, warmup_steps={}'.format(eta_min, max_steps, scheduler_config.WARMUP_STEPS))

          scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'step', 
                'frequency': 1,
            }

          return scheduler_dict

      else:
          raise ValueError(
              'Invalid scheduler name: {}'.format(scheduler_name))
      
      return scheduler

    def _log_image(self, images, mode, name, caption="", batch_idx=0):
        log_name = 'visualize/{}_{}'.format(mode, name)
        img = np.concatenate(images, axis=1)
        self.logger.experiment.log({log_name: [wandb.Image(img, caption=caption)]})
        del img

    def configure_optimizers(self):
      optimizer_name = self.config.TRAIN.OPTIMIZER
      lr = self.lr

      scheduler_config = self.config.TRAIN.SCHEDULER

      if hasattr(self, 'params') == False or len(self.params) == 0:
          return

      if optimizer_name == 'Adam':
          optimizer = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.999), weight_decay=self.config.TRAIN.WEIGHT_DECAY, amsgrad=True, eps=1e-09)
          print('using Adam optimizer: lr={}, weight_decay={}'.format(lr, self.config.TRAIN.WEIGHT_DECAY))
      elif optimizer_name == 'AdamW':
          optimizer = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.999), weight_decay=self.config.TRAIN.WEIGHT_DECAY, amsgrad=True, eps=1e-09)
          print('using AdamW optimizer: lr={}, weight_decay={}'.format(lr, self.config.TRAIN.WEIGHT_DECAY))
      elif optimizer_name == 'SGD':
          optimizer = torch.optim.SGD(self.params, lr=lr, momentum=self.config.TRAIN.MOMENTUM, weight_decay=self.config.TRAIN.WEIGHT_DECAY, nesterov=self.config.TRAIN.NESTEROV)
          print('using SGD optimizer: lr={}, momentum={}, weight_decay={}, nesterov={}'.format(lr, self.config.TRAIN.MOMENTUM, self.config.TRAIN.WEIGHT_DECAY, self.config.TRAIN.NESTEROV))
      else:
          raise ValueError(
              'Invalid optimizer name: {}'.format(optimizer_name))

      scheduler = self._configure_schedulers(optimizer, scheduler_config)

      return {'optimizer': optimizer, 'lr_scheduler': scheduler}