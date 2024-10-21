import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import radius_graph, radius, knn_graph, knn
from .base import Decoder

from prosim.core.registry import registry
from prosim.models.layers.mlp import MLP, CG_stacked
from prosim.models.layers.attention_layer import AttentionLayer
from prosim.models.layers.fourier_embedding import FourierEmbedding, FourierEmbeddingFix
from prosim.models.utils.geometry import angle_between_2d_vectors, wrap_angle

@registry.register_decoder(name='attn_fusion_relpe')
class SymCoordDecoder(Decoder):
  def _config_models(self):
    super()._config_models()
    self.attn_cfg = self.config.DECODER.ATTN
    
    if self.attn_cfg.LEARNABLE_PE:
      self.p2p_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.hidden_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
      self.s2p_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.hidden_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
    else:
      self.p2p_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.hidden_dim/4)
      self.s2p_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.hidden_dim/4)

    # prompt to prompt
    self.p2p_attn_layers = nn.ModuleList(
      [AttentionLayer(hidden_dim=self.hidden_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=False, has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
      )
  
    # scene to prompt
    self.s2p_attn_layers =  nn.ModuleList(
      [AttentionLayer(hidden_dim=self.hidden_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=True,has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
      )
  
    self.num_layers = self.attn_cfg.NUM_LAYER
    self.prompt_radius = self.attn_cfg.PROMPT_RADIUS
    self.scene_radius = self.attn_cfg.SCENE_RADIUS
    self.max_neigh = self.attn_cfg.MAX_NUM_NEIGH
  
  def _get_rel_pe(self, local_edge_index, ori_dst, pos_dst, ori_src, pos_src, mode='p2p'):
    
    ori_vec_dst = torch.stack([ori_dst.cos().squeeze(-1), ori_dst.sin().squeeze(-1)], dim=-1)
    rel_pos = pos_src[local_edge_index[0]] - pos_dst[local_edge_index[1]]

    rel_ori = wrap_angle(ori_src[local_edge_index[0]] - ori_dst[local_edge_index[1]]).squeeze(-1)
    rel_ori_vec = angle_between_2d_vectors(ctr_vector=ori_vec_dst[local_edge_index[1]], nbr_vector=rel_pos)

    if self.attn_cfg.LEARNABLE_PE:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec], dim=-1)
    else:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec, rel_ori_vec], dim=-1)

    emb_model = self.p2p_rel_pe_emb if mode == 'p2p' else self.s2p_rel_pe_emb

    rel_pe = emb_model(continuous_inputs=rel_pe_input)

    return rel_pe

  def _fusion(self, scene_emb, prompt_enc):
    # use scene context as context to decode the policy embedding
    scene_batch_idx = scene_emb['scene_batch_idx']
    scene_pos, scene_ori = scene_emb['scene_pos'], scene_emb['scene_ori']

    prompt_emb = prompt_enc['prompt_emd']
    prompt_mask = prompt_enc['prompt_mask']
    
    B = prompt_mask.shape[0]
    N = prompt_mask.shape[1]

    assert prompt_emb.shape[:2] == prompt_mask.shape

    prompt_batch_idx = torch.arange(B).unsqueeze(1).repeat(1, N).view(-1).to(prompt_emb.device)
    
    prompt_batch_idx = prompt_batch_idx[prompt_mask.view(-1)]

    prompt_emb_flat = prompt_emb.view(-1, prompt_emb.shape[-1])[prompt_mask.view(-1)]
    prompt_pos = prompt_enc['position'].view(-1, 2)[prompt_mask.view(-1)]
    prompt_ori = prompt_enc['heading'].view(-1, 1)[prompt_mask.view(-1)]
    
    # self-attention of prompts
    if self.config.REL_POS_EDGE_FUNC == 'radius':
      p2p_edge_index = radius_graph(prompt_pos, r=self.prompt_radius, batch=prompt_batch_idx, max_num_neighbors=self.max_neigh)
    else:
      p2p_edge_index = knn_graph(prompt_pos, k=self.max_neigh, batch=prompt_batch_idx)

    p2p_pe = self._get_rel_pe(p2p_edge_index, prompt_ori, prompt_pos, prompt_ori, prompt_pos, mode='p2p')

    # cross-attention from scene to prompt
    if self.config.REL_POS_EDGE_FUNC == 'radius':
      p2s_edge_index = radius(x=scene_pos, y=prompt_pos, r=self.scene_radius, batch_x=scene_batch_idx, batch_y=prompt_batch_idx, max_num_neighbors=self.max_neigh)
    else:
      p2s_edge_index = knn(x=scene_pos, y=prompt_pos, k=self.max_neigh, batch_x=scene_batch_idx, batch_y=prompt_batch_idx)
    s2p_edge_index = p2s_edge_index[[1, 0]]

    s2p_pe = self._get_rel_pe(s2p_edge_index, prompt_ori, prompt_pos, scene_ori, scene_pos, mode='s2p')

    x_s = scene_emb['scene_tokens']
    x_p = prompt_emb_flat

    for i in range(self.num_layers):
      x_p = self.p2p_attn_layers[i](x_p, p2p_pe, p2p_edge_index)
      x_p = self.s2p_attn_layers[i]((x_s, x_p), s2p_pe, s2p_edge_index)
    policy_emb = torch.zeros_like(prompt_emb)
    policy_emb[prompt_mask] = x_p
    
    return policy_emb

  def forward(self, scene_emb, prompt_enc):
    '''
    Input:
      prompt_enc: dict
        prompt_emd: [B, N, D]
        prompt_mask: [B, N]
        position: [B, N, 2]
        heading: [B, N, 1]
        agent_type: [B, N]
    
    Output:
      result: dict
        emd: [B, N, D]
        (optional) goal_prob: [B, N, 1]
        (optional) goal_point: [B, N, 1, 2]
    '''

    result = {}

    # [B, N, D]
    result['emd'] = self._fusion(scene_emb, prompt_enc)

    if self.goal_cfg.ENABLE:
      # [B, N, 1, D]
      goal_input = result['emd']
      self._goal_pred(goal_input, prompt_enc, result)

    result['agent_type'] = prompt_enc['agent_type']

    return result