import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prosim.dataset.text_utils import AGENT_TEMPLATE
from prosim.models.layers.mlp import MLP
from prosim.core.registry import registry

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LlamaTextAttn(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_dim = config.MODEL.HIDDEN_DIM
    self.llm_config = self.config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM
    self.max_txt_len = self.llm_config.MAX_TXT_LEN
    self.lora_config = self.config.MODEL.CONDITION_TRANSFORMER.TEXT_ATTN.LORA

    self.inst_prompt = "I will input scenario information and agent's embedding token for you. Please retrive and add scenario information to the agent's embedding token."

    if self.llm_config.AGENT_TOKEN_MODE == 'concat_sep':
      self.sep_prompt = "Each agent and its corresponding embedding token are listed in sequence, separated by ';', formatted as '<Agent> | [Embedding]"
    elif self.llm_config.AGENT_TOKEN_MODE == 'concat_repeat':
      self.sep_prompt = "Each agent and its corresponding embedding token are listed in sequence formatted as <Agent> [Embedding] <Agent>."
    elif self.llm_config.AGENT_TOKEN_MODE == 'concat_semantic':
      self.sep_prompt = "Each agent and its corresponding embedding token are listed in sequence, separated by ',', formatted as '<Agent> is [Embedding]."
    elif self.llm_config.REPLACE_AGENT_TOKEN:
      self.sep_prompt = "Each agent is represented with its corresponding embedding token that contains the agent's identity information and map information."

    self.prompt_text =  self.inst_prompt + " " + self.sep_prompt + "When generating embedding tokens, refer to the specific part of the scenario information text to get info for the embedding tokens." + " Input: "

    if self.llm_config.USE_SYSTEM_INSTRUCTION == False:
      self.prompt_text = ""

    self._config_models()

  def maybe_autocast(self, device, dtype=torch.bfloat16):
      # if on cpu, don't use autocast
      # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
      enable_autocast = device != torch.device("cpu")


      if enable_autocast:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.cuda.amp.autocast(dtype=dtype)
      else:
        return contextlib.nullcontext()

  def _config_models(self):
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self._config_llm()

    # prompt to llm
    self.prompt_to_llm = MLP([self.hidden_dim, self.hidden_dim, self.llm_dim], ret_before_act=True, without_norm=False)
    self.ln_prompt = nn.LayerNorm(self.llm_dim)

    # llm to condition
    self.llm_to_cond = MLP([self.llm_dim, self.hidden_dim, self.hidden_dim], ret_before_act=True, without_norm=False)

    self._config_tokenizer()

    self._config_lora()

    if self.llm_config.PROMPT_LOSS.PROMPT_MASK_PRED:
      self.prompt_mask_pred = MLP([self.hidden_dim, 1], ret_before_act=True, without_norm=True)
    
    # print('</s> embedding when init', self.llm_model.get_input_embeddings()(torch.tensor([self.llm_tokenizer.bos_token_id])))
    
  def _config_llm(self):
    self.llm_dim = 4096 # llm hidden dim for llama
    
    use_gpu = self.config.GPU is not None and len(self.config.GPU) > 0

    llm_path = self.llm_config.MODEL_PATH[self.llm_config.MODEL.upper()]
    torch_dtype = torch.bfloat16 if use_gpu else torch.float32

    print(f'loading llm from {llm_path}...')
    if 'llava' in llm_path:
      self.llm_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
      self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False, truncation_side="left")
    else:
      self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
      self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, truncation_side="left")
      print('auto device map for llm!')
    print('llm loaded!')

  def _config_lora(self):
    if self.lora_config.ENABLE:
      if self.lora_config.EMBEDDING_ONLY:
        target_modules=["embed_tokens"]
      else:
        target_modules=["q_proj", "k_proj", "v_proj", "embed_tokens"]
      
      loraconfig = LoraConfig(
        r=self.lora_config.R,
        lora_alpha=self.lora_config.ALPHA,
        target_modules=target_modules,
        lora_dropout=self.lora_config.DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
      )

      print('configuring lora with:', loraconfig)

      self.llm_model = get_peft_model(self.llm_model, loraconfig)
      self.llm_model.print_trainable_parameters()
    else:
      # forzen llm
      for name, param in self.llm_model.named_parameters():
        param.requires_grad = False
      print('all llm parameters are frozen')

  def _config_tokenizer(self):
    # we manually add special tokens to tokenizer
    self.llm_tokenizer.add_bos_token = False
    self.llm_tokenizer.add_eos_token = False

    print('Does not add any placeholder tokens to tokenizer!')
    self.llm_tokenizer.padding_side = "right"
    self.llm_tokenizer.truncation_side = 'left'

    print('Setting pad token to eos token')
    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    agent_tokens = [AGENT_TEMPLATE.format(i) for i in range(128)]
    self.llm_tokenizer.add_special_tokens({'additional_special_tokens': agent_tokens})
    self.agent_token_id_to_nidx = {self.llm_tokenizer.convert_tokens_to_ids(agent): nidx for nidx, agent in enumerate(agent_tokens)}

    self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

  def _get_llm_text_emd(self, texts, device):
    text_inputs = self.llm_tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=self.max_txt_len)
  
    text_inputs.input_ids = text_inputs.input_ids[:, 1:] # remove bos token
    text_inputs.attention_mask = text_inputs.attention_mask[:, 1:] # remove bos token

    input_ids = text_inputs.input_ids.to(device)

    text_emds = self.llm_model.get_input_embeddings()(input_ids).to(device) # (V, L, D)

    return {'llm_emd': text_emds, 'attn_mask': text_inputs.attention_mask.to(device), 'input_ids': input_ids}

  def _construct_text_prompt_batch(self, text_cond, prompt_cond_emd, prompt_mask):
    B, N, _ = prompt_cond_emd.shape
    device = prompt_cond_emd.device

    all_bidxs = torch.arange(B, device=device)[:, None].repeat(1, N) # (B, N)
    all_nidxs = torch.arange(N, device=device)[None, :].repeat(B, 1) # (B, N)

    text_mask = text_cond['mask']

    if self.llm_config.USE_TEXT_PROMPT_MASK:
      cond_mask = text_cond['prompt_mask'] & prompt_mask # (B, N)
    else:
      cond_mask = text_mask[:, None].repeat(1, N) & prompt_mask # (B, N)

    v_bidxs = all_bidxs[cond_mask] # (V)
    v_nidxs = all_nidxs[cond_mask] # (V)

    # it's possible that non of the prompt is explicitly mentioned in the text.
    # if USE_TEXT_PROMPT_MASK == True, do not use this text
    batch_cond_mask = cond_mask.any(dim=1) # (B)
    T = batch_cond_mask.sum().item() # number of valid texts
    
    P = prompt_mask[text_mask].sum(dim=1).max().item() # max number of prompts for valid texts
    
    t_p_emd = torch.zeros(T, P, self.hidden_dim, device=device) # (T, P, hidden_dim)
    t_mask = torch.zeros(T, P, device=device, dtype=torch.bool) # (T, P)
    t_nidxs = torch.ones(T, P, device=device, dtype=torch.long) * -1 # (T, P)
    t_bidxs = torch.unique(v_bidxs)[:, None].repeat(1, P) # (T, P)
    
    for i, bidx in enumerate(t_bidxs[:, 0]):
      nidxs = v_nidxs[v_bidxs == bidx]
      t_p_emd[i, :len(nidxs)] = prompt_cond_emd[bidx, nidxs]
      t_mask[i, :len(nidxs)] = True
      t_nidxs[i, :len(nidxs)] = nidxs
    
    if self.llm_config.DETACH_PROMPT_TOKEN:
      t_p_emd = t_p_emd.detach()
      print('\t\tDEBUG: prompt token is detached')
    
    if not self.llm_config.USE_PROMPT_TOKEN:
      t_p_emd = t_p_emd * 0.0
      print('\t\tDEBUG: prompt token is zeroed out')

    # convert prompt emd to llm emd
    t_p_emd_llm = self.prompt_to_llm(t_p_emd) # (T, P, D)
    t_p_emd_llm = self.ln_prompt(t_p_emd_llm) # (T, P, D)
    
    return t_p_emd_llm, t_bidxs, t_nidxs, t_mask

  def _get_prompt_agent_name_emd(self, t_nidx, t_mask):
    '''
    input:
      t_nidx (tensor): (T, P)
      t_mask (tensor): (T, P)
    
    output:
      agent_token_emd (tensor): (T, P, D)
      agent_token_id (tensor): (T, P)
    '''
    T, P = t_nidx.shape

    # get valid agent name strings from template
    valid_t_nidx = t_nidx[t_mask] # (V)
    valid_agent_names = [AGENT_TEMPLATE.format(nidx) for nidx in valid_t_nidx.tolist()] # (V)

    # get llm emd for valid agent names
    v_token_dict = self._get_llm_text_emd(valid_agent_names, device=t_mask.device)
    v_token_emd = v_token_dict['llm_emd'] # (V, 1, D)
    v_token_ids = v_token_dict['input_ids'] # (V, 1)
    
    if v_token_emd.shape[1] != 1:
      print(valid_agent_names)
    
    assert v_token_emd.shape[1] == 1

    all_token_emd = torch.zeros(T, P, self.llm_dim, device=t_nidx.device) # (T, P, D)
    all_token_ids = torch.ones(T, P, device=t_nidx.device, dtype=torch.long) * self.llm_tokenizer.eos_token_id # (T, P)

    dtype = all_token_emd.dtype
    
    all_token_emd[t_mask] = v_token_emd[:, 0].to(dtype)
    all_token_ids[t_mask] = v_token_ids[:, 0]

    return all_token_emd, all_token_ids

  def _add_bos_eos_tokens(self, llm_input, llm_ids, attn_mask, device):
    T, L, D = llm_input.shape
    device = llm_input.device

    bos_token_id = torch.tensor([self.llm_tokenizer.bos_token_id], device=device)
    eos_token_id = torch.tensor([self.llm_tokenizer.eos_token_id], device=device)

    bos_token_emd = self.llm_model.get_input_embeddings()(bos_token_id)[None, :].repeat(T, 1, 1) # (T, 1, D)
    eos_token_emd = self.llm_model.get_input_embeddings()(eos_token_id)[None, :].repeat(T, 1, 1) # (T, 1, D)
    bos_token_mask = torch.ones(T, 1, device=device, dtype=torch.bool) # (T, 1)
    eos_token_mask = torch.ones(T, 1, device=device, dtype=torch.bool) # (1)

    llm_input = torch.cat([bos_token_emd, llm_input, eos_token_emd], dim=1) # (T, L+P+2, D)
    attn_mask = torch.cat([bos_token_mask, attn_mask, eos_token_mask], dim=1) # (T, L+P+2)

    llm_ids = torch.cat([bos_token_id[None, :].repeat(T, 1), llm_ids, eos_token_id[None, :].repeat(T, 1)], dim=1) # (T, L+P+2)

    return llm_input, llm_ids, attn_mask

  def _concate_prompt_name_emd(self, t_n_emd_llm, t_n_ids, t_p_emd_llm, t_p_ids, t_mask):
    T, P, D = t_n_emd_llm.shape
    device = t_n_emd_llm.device
    dtype = t_n_emd_llm.dtype

    if self.llm_config.AGENT_TOKEN_MODE == 'add':
      prompt_emd = t_n_emd_llm + t_p_emd_llm # (T, P, D)
      prompt_lidx = torch.arange(0, P, device=device)[None, :].repeat(T, 1)# (T, P)
      prompt_mask = t_mask # (T, P)
      prompt_ids = t_n_ids # (T, P)
    
    elif self.llm_config.AGENT_TOKEN_MODE == 'none':
      prompt_emd = t_p_emd_llm
      prompt_lidx = torch.arange(0, P, device=device)[None, :].repeat(T, 1)# (T, P)
      prompt_mask = t_mask # (T, P)
      prompt_ids = t_n_ids # (T, P)

    elif self.llm_config.AGENT_TOKEN_MODE == 'concat':
      prompt_emd = torch.empty(T, 2 * P, D, dtype=dtype, device=device)
      prompt_emd[:, 0::2, :] = t_n_emd_llm # (T, 2P, D)
      prompt_emd[:, 1::2, :] = t_p_emd_llm # (T, 2P, D)

      prompt_mask = torch.empty(T, 2 * P, dtype=torch.bool, device=device)
      prompt_mask[:, 0::2] = t_mask
      prompt_mask[:, 1::2] = t_mask
      
      prompt_lidx = torch.arange(1, 2 * P, 2)[None, :].repeat(T, 1) # (T, P)

      prompt_ids = torch.empty(T, 2 * P, dtype=torch.long, device=device)
      prompt_ids[:, 0::2] = t_n_ids
      prompt_ids[:, 1::2] = t_p_ids

    elif self.llm_config.AGENT_TOKEN_MODE == 'concat_repeat':
      prompt_emd = torch.empty(T, 3 * P, D, dtype=dtype, device=device)
      prompt_emd[:, 0::3, :] = t_n_emd_llm
      prompt_emd[:, 1::3, :] = t_p_emd_llm
      prompt_emd[:, 2::3, :] = t_n_emd_llm

      prompt_mask = torch.empty(T, 3 * P, dtype=torch.bool, device=device)
      prompt_mask[:, 0::3] = t_mask
      prompt_mask[:, 1::3] = t_mask
      prompt_mask[:, 2::3] = t_mask

      prompt_lidx = torch.arange(1, 3 * P, 3)[None, :].repeat(T, 1) # (T, P)

      prompt_ids = torch.empty(T, 3 * P, dtype=torch.long, device=device)
      prompt_ids[:, 0::3] = t_n_ids
      prompt_ids[:, 1::3] = t_p_ids
      prompt_ids[:, 2::3] = t_n_ids

    elif self.llm_config.AGENT_TOKEN_MODE in ['concat_sep', 'concat_semantic']:

      if self.llm_config.AGENT_TOKEN_MODE == 'concat_sep':
        # <A0> | [Emd] ; <A1> | [A1_Emd] | ; ...
        sep_1_ids = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids('|')], device=device)[None, :].repeat(T, P) # (T, P)
        sep_2_ids = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids(';')], device=device)[None, :].repeat(T, P) # (T, P)
      elif self.llm_config.AGENT_TOKEN_MODE == 'concat_semantic':
        # <A0> is [Emd], <A2> is [A2_Emd] ; ...
        sep_1_ids = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids('is')], device=device)[None, :].repeat(T, P) # (T, P)
        sep_2_ids = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids(',')], device=device)[None, :].repeat(T, P) # (T, P)

      sep_1_emd = self.llm_model.get_input_embeddings()(sep_1_ids) # (T, P, D)
      sep_2_emd = self.llm_model.get_input_embeddings()(sep_2_ids) # (T, P, D)

      prompt_emd = torch.empty(T, 4 * P, D, dtype=dtype, device=device)
      prompt_emd[:, 0::4, :] = t_n_emd_llm
      prompt_emd[:, 1::4, :] = sep_1_emd
      prompt_emd[:, 2::4, :] = t_p_emd_llm
      prompt_emd[:, 3::4, :] = sep_2_emd

      prompt_mask = torch.empty(T, 4 * P, dtype=torch.bool, device=device)
      prompt_mask[:, 0::4] = t_mask
      prompt_mask[:, 1::4] = t_mask
      prompt_mask[:, 2::4] = t_mask
      prompt_mask[:, 3::4] = t_mask
      
      prompt_lidx = torch.arange(2, 4 * P, 4)[None, :].repeat(T, 1) # (T, P)

      prompt_ids = torch.empty(T, 4 * P, dtype=torch.long, device=device)
      prompt_ids[:, 0::4] = t_n_ids
      prompt_ids[:, 1::4] = sep_1_ids
      prompt_ids[:, 2::4] = t_p_ids
      prompt_ids[:, 3::4] = sep_2_ids

    return prompt_emd, prompt_ids, prompt_lidx, prompt_mask

  def _concat_prompt_text_emd(self, text_emd_dict, t_p_emd, t_n_emd_llm, t_n_ids, t_mask, add_special_tokens):
    '''
    concatenate text emd and prompt emd
    output:
      llm_input (tensor): (T, L+P, D)
      attn_mask (tensor): (T, L+P)
      prompt_lidx (tensor): (T, P)
      llm_input_ids (tensor): (T, L+P)
    '''
    text_llm_emd = text_emd_dict['llm_emd'] # (T, L, D)
    text_attn_mask = text_emd_dict['attn_mask'] # (T, L)
    T, L, D = text_llm_emd.shape
    P = t_p_emd.shape[1]
    device = t_p_emd.device
    dtype = t_p_emd.dtype

    t_p_ids = torch.ones(T, P, device=device, dtype=torch.long) * self.llm_tokenizer.eos_token_id # (T, P)
    prompt_emd, prompt_ids, prompt_lidx, prompt_mask = self._concate_prompt_name_emd(t_n_emd_llm, t_n_ids, t_p_emd, t_p_ids, t_mask)

    if self.llm_config.PROMPT_TAIL:
      llm_input = torch.cat([text_llm_emd, prompt_emd], dim=1) # (T, L+P, D)
      attn_mask = torch.cat([text_attn_mask, prompt_mask], dim=1) # (T, L+P)
      llm_ids = torch.cat([text_emd_dict['input_ids'], prompt_ids], dim=1) # (T, L+P)
      prompt_lidx = prompt_lidx + L
    else:
      llm_input = torch.cat([prompt_emd, text_llm_emd], dim=1) # (T, L+P, D)
      attn_mask = torch.cat([prompt_mask, text_attn_mask], dim=1) # (T, L+P)
      llm_ids = torch.cat([prompt_ids, text_emd_dict['input_ids']], dim=1) # (T, L+P)

    if add_special_tokens:
      # add special tokens
      llm_input, llm_ids, attn_mask = self._add_bos_eos_tokens(llm_input, llm_ids, attn_mask, device)
      prompt_lidx += 1
    
    return llm_input, llm_ids, attn_mask, prompt_lidx

  def _obtain_llm_input_with_prompt(self, t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond, add_special_tokens):
    text_list = [text_cond['input'][bidx] for bidx in t_bidxs[:, 0].tolist()] # (T)
    device = t_p_emd.device

    text_emd_dict = self._get_llm_text_emd(text_list, device) # {'llm_emd': (T, L, D), 'attn_mask': (T, L)}

    t_n_emd_llm, t_n_ids = self._get_prompt_agent_name_emd(t_nidxs, t_mask) # (T, P, D)

    llm_input, llm_ids, attn_mask, prompt_lidx = self._concat_prompt_text_emd(text_emd_dict, t_p_emd, t_n_emd_llm, t_n_ids, t_mask, add_special_tokens) # (T, L+P(+2/+0), D), (T, L+P(+2/+0), (T, P)

    return llm_input, llm_ids, attn_mask, prompt_lidx

  def _replace_agent_token_with_prompt_emd(self, llm_input, llm_ids, prompt_emd, prompt_nidxs):
    '''
    replace agent token with corresponding prompt emd
    output:
      llm_input (tensor): (T, L+P, D)
      llm_ids (tensor): (T, L+P)
    '''
    T, L, D = llm_input.shape
    P = prompt_emd.shape[1]
    device = llm_input.device

    for tidx in range(T):
      for lidx in range(L):
        token_id = llm_ids[tidx, lidx].item()
        
        # if token_id is agent token, replace with prompt emd
        if token_id in self.agent_token_id_to_nidx:
          nidx = self.agent_token_id_to_nidx[token_id]
          
          assert nidx in prompt_nidxs[tidx]
          pidx = torch.nonzero(prompt_nidxs[tidx] == nidx)[0].item()

          llm_input[tidx, lidx] = prompt_emd[tidx, pidx]

          # (keep llm_ids unchanged for visualization purpose)
          # llm_ids[tidx, lidx] = self.llm_tokenizer.eos_token_id
    
    return llm_input, llm_ids

  def _query_prompt_cond_from_llm(self, t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond):
    '''
    input:
      t_p_emd (tensor): (T, P, hidden_dim)
      t_bidxs (tensor): (T, P)
      t_nidxs (tensor): (T, P)
      t_mask (tensor): (T, P)
    output:
      llm_emd (tensor): (T, P, D)
    '''

    add_special_tokens = self.llm_config.ADD_BOS_EOS
    llm_input, llm_ids, attn_mask, prompt_lidx = self._obtain_llm_input_with_prompt(t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond, add_special_tokens) # (T, L+P+2, D), (T, L+P+2), (T, P)
    
    T, P = t_p_emd.shape[:2]
    device = t_p_emd.device
    dtype = t_p_emd.dtype

    # grad_context = torch.enable_grad() if self.lora_config.ENABLE else torch.no_grad()

    if self.llm_config.REPLACE_AGENT_TOKEN:
      # replace agent token with prompt emd
      llm_input, llm_ids = self._replace_agent_token_with_prompt_emd(llm_input, llm_ids, t_p_emd, t_nidxs)
    # with grad_context:
    with self.maybe_autocast(device):
      output = self.llm_model(
        inputs_embeds=llm_input,
        attention_mask=attn_mask,
        output_hidden_states=True,
        return_dict=True,
        output_attentions=False
      )
    
    # print('DEBUG: saving llm output for visualization')

    hidden_state = output.hidden_states[-1].to(dtype).to(device) # (T, L+P+2, D)
    prompt_tidx = torch.arange(T, device=device)[:, None].repeat(1, P) # (T, P)
    llm_emd = hidden_state[prompt_tidx, prompt_lidx, :] # (T, P, D)



    return llm_emd

  def _compute_prompt_loss(self, v_cond_emd, v_bidxs, v_nidx, text_cond):
    '''
      compute prompt-level loss to facilitate llm training

      input:
        v_cond_emd (tensor): (V, hidden_dim)
        v_bidxs (tensor): (V) 
        v_nidx (tensor): (V)
        text_cond (dict):
          'input': [B] list of strings
          'mask': [B] (tensor): 1 for valid string, 0 for padding
          'prompt_mask': [B, N] (tensor): 1 for valid prompt, 0 for padding
      
      output:
        prompt_loss (dict): {'prompt_mask_pred_loss': loss}

    '''
    prompt_loss = {}

    # compute prompt_mask prediction loss
    if self.llm_config.PROMPT_LOSS.PROMPT_MASK_PRED:
      assert self.llm_config.USE_TEXT_PROMPT_MASK == False, "prompt mask prediction is invalid when explitly using prompt mask"

      v_prompt_mask = text_cond['prompt_mask'][v_bidxs, v_nidx].to(torch.float) # (V)

      v_prompt_mask_pred = self.prompt_mask_pred(v_cond_emd).squeeze(1) # (V)

      prompt_loss['prompt_mask_pred_loss'] = F.binary_cross_entropy_with_logits(v_prompt_mask_pred, v_prompt_mask)

    return prompt_loss

  def forward(self, text_cond, prompt_cond_emd, prompt_mask, **kwargs):
    '''
      text_cond (dict):
        'input': [B] list of strings
        'mask': [B] (tensor): 1 for valid string, 0 for padding
      
      prompt_cond_emd (tensor): (B, N, hidden_dim)
      prompt_mask (tensor): (B, N)
    '''
    # print(text_cond['input'])
    # print(text_cond['mask'])

    # if no valid text, return prompt condition emd
    if text_cond['mask'].sum().item() == 0:
      print('no valid text')
      return prompt_cond_emd, None

    # construct text prompt batch to query llm with text
    t_p_emd, t_bidxs, t_nidxs, t_mask = self._construct_text_prompt_batch(text_cond, prompt_cond_emd, prompt_mask)

    if t_mask.sum().item() == 0:
      print('no text mentioning agent')
      print('text:', text_cond['input'])
      return prompt_cond_emd, None

    for b in range(t_p_emd.shape[0]):
      text_cond['input'][b] = text_cond['input'][b] + ' ' + self.prompt_text

    # try:
    # query llm with text and prompt condition
    t_cond_emd_llm = self._query_prompt_cond_from_llm(t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond) # (T, D)

    # convert llm emd to condition emd
    v_cond_emd_llm = t_cond_emd_llm[t_mask] # (V, D)
    v_cond_emd = self.llm_to_cond(v_cond_emd_llm) # (V, hidden_dim)

    # add text condition emd as residual to prompt condition emd
    v_bidxs = t_bidxs[t_mask] # (V)
    v_nidx = t_nidxs[t_mask] # (V)
    # prompt_cond_emd[v_bidxs, v_nidx] += v_cond_emd
    prompt_cond_emd[v_bidxs, v_nidx] += v_cond_emd

    prompt_loss = self._compute_prompt_loss(v_cond_emd, v_bidxs, v_nidx, text_cond)


    return prompt_cond_emd, prompt_loss
    
class LlamaTextAttnQA(LlamaTextAttn):
  def __init__(self, config):
    self.qa_cfg = config.MODEL.CONDITION_TRANSFORMER.TEXT_ATTN.QA
    super().__init__(config)

    self.inst_prompt = "I will input agent's embedding token for you. Please retrive information for agent's embedding token."
    if self.llm_config.AGENT_TOKEN_MODE == 'concat_sep':
      self.sep_prompt = " Each agent and its corresponding embedding token are listed in sequence, separated by ';', formatted as '<Agent> | [Embedding];'. When generating embedding tokens, refer to the corresponding agent's embedding token to get information"
    elif self.llm_config.AGENT_TOKEN_MODE == 'concat_repeat':
      self.sep_prompt = " Each agent and its corresponding embedding token are listed in sequence formatted as '<Agent> [Embedding] <Agent>'. When generating embedding tokens, refer to the corresponding agent's embedding token to get information"
    elif self.llm_config.AGENT_TOKEN_MODE == 'concat_semantic':
      self.sep_prompt = " Each agent and its corresponding embedding token are listed in sequence, separated by ',', formatted as '<Agent> is [Embedding],'. When generating embedding tokens, refer to the corresponding agent's embedding token to get information"
    else: 
      self.sep_prompt = ""
    
    self.prompt_text = self.inst_prompt + self.sep_prompt + " Input: "

  def _config_models(self):
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self._config_llm()

    # prompt to llm
    self.prompt_to_llm = MLP([self.hidden_dim, self.hidden_dim, self.llm_dim], ret_before_act=True, without_norm=False)
    self.ln_prompt = nn.LayerNorm(self.llm_dim)

    if self.qa_cfg.GT_QUERY:
      self.gt_to_hidden = MLP([2, self.hidden_dim], ret_before_act=True, without_norm=False)

    self._config_tokenizer()
    self._config_lora()

  def _prepare_qa_text(self, batch):
    question_type = self.qa_cfg.QUESTION_TYPE

    batch_data = batch.extras['io_pairs_batch'][question_type][:, 0]
    B, N, _ = batch_data.shape

    questions = []
    answers = []
    nidxs = []

    for b in range(B):
      valid_mask = batch.extras['io_pairs_batch']['mask'][b, 0]
      valid_nidxs = torch.where(valid_mask)[0]
      random_nidx = valid_nidxs[torch.randint(len(valid_nidxs), (1,)).item()].item()
      gt_data = batch_data[b, random_nidx]

      question = f' Question: {question_type} of agent {AGENT_TEMPLATE.format(random_nidx)} is?'
      
      if self.qa_cfg.CONTEXTURAL_QUESTION:
        question += f' given embedding of {AGENT_TEMPLATE.format(random_nidx)} |'

      answer = f'Answer:({gt_data[0].item():.2f}, {gt_data[1].item():.2f})'

      questions.append(question)
      answers.append(answer)
      nidxs.append(random_nidx)
    
    return questions, answers, nidxs

  def _get_full_input_and_label(self, input_emd, input_ids, input_mask, question_emd_dict, answer_emd_dict, device):

    full_input_emd = torch.cat([input_emd, question_emd_dict['llm_emd'], answer_emd_dict['llm_emd']], dim=1)
    full_input_mask = torch.cat([input_mask, question_emd_dict['attn_mask'], answer_emd_dict['attn_mask']], dim=1)
    full_input_ids = torch.cat([input_ids, question_emd_dict['input_ids'], answer_emd_dict['input_ids']], dim=1)

    full_input_emd, full_input_ids, full_input_mask = self._add_bos_eos_tokens(full_input_emd, full_input_ids, full_input_mask, device)

    question_size = question_emd_dict['llm_emd'].shape[1]
    answer_size = answer_emd_dict['llm_emd'].shape[1]

    B, T = full_input_emd.shape[:2]

    labels = torch.full((B, T), -100, device=device, dtype=torch.long)
    labels[:, -answer_size-1:-1] = answer_emd_dict['input_ids']

    return full_input_emd, full_input_ids, full_input_mask, labels, question_size, answer_size
  

  def forward(self, text_cond, prompt_cond_emd, prompt_mask, **kwargs):
    '''
      text_cond (dict):
        'input': [B] list of strings
        'mask': [B] (tensor): 1 for valid string, 0 for padding
      
      prompt_cond_emd (tensor): (B, N, hidden_dim)
      prompt_mask (tensor): (B, N)
    '''
    # construct text prompt batch to query llm with text
    
    if self.qa_cfg.GT_QUERY:
      question_type = self.qa_cfg.QUESTION_TYPE
      gt_data = kwargs['batch'].extras['io_pairs_batch'][question_type][:, 0] # (B, N, 2)
      prompt_cond_emd = self.gt_to_hidden(gt_data)

    t_p_emd, t_bidxs, t_nidxs, t_mask = self._construct_text_prompt_batch(text_cond, prompt_cond_emd, prompt_mask)
    device = t_p_emd.device

    if self.qa_cfg.QUESTION_TYPE != 'goal':
      for b in range(t_p_emd.shape[0]):
        text_cond['input'][b] = self.prompt_text

    # append prompt tokens along with text input tokens
    input_emd, input_ids, input_mask, prompt_lidx = self._obtain_llm_input_with_prompt(t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond, False) # (T, L+P, D), (T, L+P)

    # get question and answer strs
    questions, answers, nidxs = self._prepare_qa_text(kwargs['batch'])
    question_emd_dict = self._get_llm_text_emd(questions, device)
    if self.qa_cfg.CONTEXTURAL_QUESTION:
      context_emds = []
      for b in range(input_emd.shape[0]):
        nidx = nidxs[b]
        pidx = t_nidxs[b].cpu().tolist().index(nidx)
        lidx = prompt_lidx[b, pidx].item()
        context_emds.append(input_emd[b, lidx, :])
      context_emds = torch.stack(context_emds)[:, None] # (T, 1, D)
      question_emd_dict['llm_emd'] = torch.cat([question_emd_dict['llm_emd'], context_emds], dim=1)
      question_emd_dict['attn_mask'] = torch.cat([question_emd_dict['attn_mask'], torch.ones(input_emd.shape[0], 1, device=device, dtype=torch.bool)], dim=1)
      question_emd_dict['input_ids'] = torch.cat([question_emd_dict['input_ids'], torch.ones(input_emd.shape[0], 1, device=device, dtype=torch.long) * self.llm_tokenizer.eos_token_id], dim=1)

    answer_emd_dict = self._get_llm_text_emd(answers, device)

    # get full llm input and labels
    full_input_emd, full_input_ids, full_input_mask, labels, question_size, answer_size = self._get_full_input_and_label(input_emd, input_ids, input_mask, question_emd_dict, answer_emd_dict, device)

    if self.llm_config.REPLACE_AGENT_TOKEN:
      full_input_emd, full_input_ids = self._replace_agent_token_with_prompt_emd(full_input_emd, full_input_ids, t_p_emd, t_nidxs)

    with self.maybe_autocast(device):
      output = self.llm_model(
        inputs_embeds=full_input_emd,
        attention_mask=full_input_mask,
        output_hidden_states=False,
        labels=labels,
        return_dict=True,
        output_attentions=False,
        # output_attentions=True,
      )
    
    print('qa label: ', questions[0] + answers[0])
    qa_logits = output['logits'][0, -(answer_size+2):, :]
    print('qa output: ', self.llm_tokenizer.decode(torch.argmax(qa_logits, dim=-1)))
    return {'loss': output['loss']}, None
    
text_attns = {'llama': LlamaTextAttn, 'llama_qa': LlamaTextAttnQA}