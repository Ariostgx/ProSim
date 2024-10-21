import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from prosim.models.layers.mlp import MLP

from transformers import LlamaTokenizer, LlamaForCausalLM

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class LLMEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_dim = config.MODEL.HIDDEN_DIM
    self.llm_config = self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM
    self.max_txt_len = self.llm_config.MAX_TXT_LEN

    self._config_models()

  def maybe_autocast(self, device, dtype=torch.float16):
      # if on cpu, don't use autocast
      # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
      enable_autocast = device != torch.device("cpu")
      enable_autocast = True

      if enable_autocast:
          return torch.cuda.amp.autocast(dtype=dtype)
      else:
          return contextlib.nullcontext()

  def _config_models(self):
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    llm_dim = self.llm_config.HIDDEN_DIM
    use_gpu = self.config.GPU is not None and len(self.config.GPU) > 0

    if self.llm_config.USE_PROMPT_TOKEN:
       self.prompt_to_llm_emd = MLP([self.hidden_dim, self.hidden_dim, llm_dim], ret_before_act=True, without_norm=True)

    self.llm_to_cond_emd = MLP([llm_dim, self.hidden_dim, self.hidden_dim], ret_before_act=True, without_norm=True)

    llm_path = self.llm_config.MODEL_PATH[self.llm_config.MODEL.upper()]

    self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False, truncation_side="left")
    torch_dtype = torch.float16 if use_gpu else torch.float32

    self.llm_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    # forzen llm
    for name, param in self.llm_model.named_parameters():
      param.requires_grad = False
    
    self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
    self.llm_tokenizer.padding_side = "right"
    self.llm_tokenizer.truncation_side = 'left'

    self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

  def _get_llm_text_emd(self, texts: list[str], device: str = 'cuda'):
    text_inputs = self.llm_tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=self.max_txt_len)

    text_emds = self.llm_model.get_input_embeddings()(text_inputs.input_ids.to(device)) # (V, T, D)

    return {'llm_input': text_emds, 'attn_mask': text_inputs.attention_mask.to(device)}

  def _get_llm_feature(self, llm_input_emd, llm_input_mask):
    device = llm_input_emd.device

    with self.maybe_autocast(device):
      with torch.no_grad():
        output = self.llm_model(
          inputs_embeds=llm_input_emd,
          attention_mask=llm_input_mask,
          output_hidden_states=True,
          return_dict=True,
        )

    hidden_state = output.hidden_states[-1] # (V, T, D)

    seq_idx = (torch.sum(llm_input_mask, dim=1) - 1).to(torch.long)
    b_idx = torch.arange(seq_idx.size(0)).to(hidden_state.device)
    llm_emd = hidden_state[b_idx, seq_idx, :] # (V, D)
    return llm_emd

  def _append_prompt_token(self, llm_input_dict, prompt_emd):
    llm_inputs = []
    llm_attn_masks = []

    device = llm_input_dict['attn_mask'].device

    for vid in range(prompt_emd.shape[0]):
      token_cnt = llm_input_dict['attn_mask'][vid].sum().item()
      llm_inputs.append(
        torch.cat([
          llm_input_dict['llm_input'][vid, :token_cnt],
          prompt_emd[vid].unsqueeze(0),
          llm_input_dict['llm_input'][vid, token_cnt:]
        ], dim=0)
      )
      llm_attn_masks.append(
        torch.cat([
          llm_input_dict['attn_mask'][vid, :token_cnt],
          torch.ones(1, device=device),
          llm_input_dict['attn_mask'][vid, token_cnt:]
        ], dim=0)
      )

    result = {}

    result['llm_input'] = torch.stack(llm_inputs, dim=0)
    result['attn_mask'] = torch.stack(llm_attn_masks, dim=0)

    return result

  def forward(self, valid_texts, bidxs, nidxs, **kwargs):
    '''
    valid_texts: list of strings, each string is a valid text (size V)
    '''
    device = bidxs.device

    llm_input_dict = self._get_llm_text_emd(valid_texts, device)

    if self.llm_config.USE_PROMPT_TOKEN:
      all_prompt_emd = kwargs['prompt_emd'] # (B, N, hidden_dim)
      prompt_emd = all_prompt_emd[bidxs, nidxs] # (V, hidden_dim)
      prompt_emd = self.prompt_to_llm_emd(prompt_emd) # (V, D)

      llm_input_dict = self._append_prompt_token(llm_input_dict, prompt_emd)
    
    llm_emd = self._get_llm_feature(llm_input_dict['llm_input'], llm_input_dict['attn_mask']) # (V, D)

    cond_emds = self.llm_to_cond_emd(llm_emd.to(torch.float32)) # (V, hidden_dim)

    return cond_emds