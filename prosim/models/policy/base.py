import torch.nn as nn

from prosim.core.registry import registry
from .temporal_ar import PolicyNoRNN

act_decoders = {'policy_no_rnn': PolicyNoRNN}

@registry.register_policy(name='rel_pe_temporal')
class Policy_RelPE_Temporal(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.p_config = config.MODEL.POLICY
    self._config_models()
  
  def _config_models(self):
    self.act_decoder = act_decoders[self.p_config.ACT_DECODER.TYPE](self.config)

  def forward(self, policy_emd, batch_obs, batch_map, batch_pos, pair_names, latent_state):
    return self.act_decoder(policy_emd, batch_obs, batch_map, batch_pos, pair_names, latent_state)

  def format_latent_state(self, lante_state_dict, all_batch_pair_names):
    return self.act_decoder.format_latent_state(lante_state_dict, all_batch_pair_names)
