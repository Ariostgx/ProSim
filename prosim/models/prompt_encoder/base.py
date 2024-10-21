import torch.nn as nn
import torch.nn.functional as F

from prosim.models.layers.mlp import MLP
from prosim.core.registry import registry
@registry.register_prompt_encoder(name='agent_status')
class PromptEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self._config_models()
    self.state_encoder.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

  def _config_prompt_models(self):
    status_cfg = self.config.PROMPT.AGENT_STATUS
    input_dim = 0
    if status_cfg.USE_VEL:
      input_dim += 2
    if status_cfg.USE_EXTEND:
      input_dim += 2
    if status_cfg.USE_AGENT_TYPE:
      input_dim += 3
    self.state_encoder = MLP([input_dim, self.config.MODEL.HIDDEN_DIM, self.config.MODEL.HIDDEN_DIM], ret_before_act=True)

  def _config_models(self):
    self._config_prompt_models()

  def _prompt_encode(self, prompt_input):
    prompt = prompt_input['prompt']
    prompt_mask = prompt_input['prompt_mask']
    device = next(self.parameters()).device
    
    prompt = prompt.to(device)
    prompt_mask = prompt_mask.to(device)

    prompt_emd = self.state_encoder(prompt)
    return prompt_emd, prompt_mask

  def forward(self, prompt_input):
    prompt_emd, prompt_mask = self._prompt_encode(prompt_input)
    
    return prompt_emd, prompt_mask