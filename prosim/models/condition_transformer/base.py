import torch.nn as nn

from .condition_encoders import condition_encoders
from .condition_attns import condition_attns
from .text_attns import text_attns
class ConditionTransformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.use_pe = self.config.MODEL.CONDITION_TRANSFORMER.PE.ENABLE
    self._config_cond_types()
    self._config_models()
  
  def _config_cond_types(self):
    cond_types = self.config.PROMPT.CONDITION.TYPES
    self.cond_types = [type_name for type_name in cond_types if 'OneText' not in type_name]
    self.text_types = [type_name for type_name in cond_types if 'OneText' in type_name]

  def _config_models(self):
    self._config_condition_encoders()
    self._config_condition_attn()
  
  def _config_condition_encoders(self):
    if len(self.cond_types) > 0:
      self.condition_encoders = nn.ModuleDict()
      for cond_type in self.cond_types:
        self.condition_encoders[cond_type] = condition_encoders[cond_type](self.config)
  
  def _config_condition_attn(self):
    if len(self.cond_types) > 0:
      model_type = self.config.MODEL.CONDITION_TRANSFORMER.ATTN_TYPE
      self.condition_attn = condition_attns[model_type](self.config)
    
    if len(self.text_types) > 0:
      model_type = self.config.MODEL.CONDITION_TRANSFORMER.TEXT_ATTN.TYPE
      self.text_attn = text_attns[model_type](self.config)
  
  def forward(self, condition_data, **kwargs):
    condition_emds = {}

    # apply condition encoders for fixed prompt_idx condition types (non-text)
    if len(self.cond_types) > 0:
      for cond_type, cond_encoder in self.condition_encoders.items():
        if cond_type in condition_data.keys() and condition_data[cond_type]['input'].shape[1] > 0:
          cond_emd_dicts = cond_encoder(condition_data[cond_type], **kwargs)
          condition_emds.update(cond_emd_dicts)
      
      prompt_condition_emd = self.condition_attn(condition_emds=condition_emds, **kwargs)
    
    else:
      prompt_condition_emd = kwargs['prompt_emd']
    
    # apply condition encoders for text prompt_idx condition types
    if len(self.text_types) > 0 and self.text_types[0] in condition_data.keys():
      text_cond = condition_data[self.text_types[0]]
      prompt_condition_emd, prompt_loss = self.text_attn(text_cond, prompt_condition_emd, **kwargs)
    else:
      print('No condition data for text condition transformer')
      prompt_loss = None
    
    return prompt_condition_emd, prompt_loss