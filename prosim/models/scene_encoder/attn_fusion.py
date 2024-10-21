import torch
import torch.nn as nn

from prosim.models.layers.mlp import MLP
from torch_cluster import radius_graph, radius, knn_graph

from .base import SceneEncoder
from prosim.core.registry import registry
from prosim.models.layers.attention_layer import AttentionLayer
from prosim.models.layers.fourier_embedding import FourierEmbedding, FourierEmbeddingFix
from prosim.models.utils.geometry import angle_between_2d_vectors, wrap_angle
@registry.register_scene_encoder(name='attn_fusion_relpe')
class AttentionSceneEncoderRelPE(SceneEncoder):
  def _config_models(self):
    super()._config_models()
    self.obs_update_cfg = self.config.MODEL.OBS_UPDATE

    if self.obs_update_cfg.FUSION == 'mlp':
      self.obs_update_mlp = MLP([self.hidden_dim * 2, self.hidden_dim, self.hidden_dim], ret_before_act=True)
  
  def _config_fusion(self):
    self.attn_cfg = self.model_cfg.ATTN

    if self.attn_cfg.LEARNABLE_PE:
      self.a2a_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.hidden_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
      self.s2s_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.hidden_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
    else:
      self.a2a_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.hidden_dim/4)
      self.s2s_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.hidden_dim/4)

    self.a2a_attn_layers = nn.ModuleList(
        [AttentionLayer(hidden_dim=self.hidden_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=False, has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
        )
  
    self.s2s_attn_layers =  nn.ModuleList(
        [AttentionLayer(hidden_dim=self.hidden_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=False, has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
        )
  
    self.num_layers = self.attn_cfg.NUM_LAYER
    self.max_neigh = self.attn_cfg.MAX_NUM_NEIGH
    self.agent_radius = self.attn_cfg.AGENT_RADIUS
    self.scene_radius = self.attn_cfg.SCENE_RADIUS

  def _get_rel_pe(self, local_edge_index, token_ori, token_pos, mode='s2s'):
    orient_vector_pl = torch.stack([token_ori.cos().squeeze(-1), token_ori.sin().squeeze(-1)], dim=-1)
    rel_pos = token_pos[local_edge_index[0]] - token_pos[local_edge_index[1]]
    rel_ori = wrap_angle(token_ori[local_edge_index[0]] - token_ori[local_edge_index[1]]).squeeze(-1)
    rel_ori_vec = angle_between_2d_vectors(ctr_vector=orient_vector_pl[local_edge_index[1]], nbr_vector=rel_pos)

    if self.attn_cfg.LEARNABLE_PE:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec], dim=-1)
    else:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec, rel_ori_vec], dim=-1)

    emb_model = self.s2s_rel_pe_emb if mode == 's2s' else self.a2a_rel_pe_emb
    
    rel_pe = emb_model(continuous_inputs=rel_pe_input)
    return rel_pe

  def _get_rel_pe_src2dst(self, local_edge_index, ori_dst, pos_dst, ori_src, pos_src):
    ori_vec_dst = torch.stack([ori_dst.cos().squeeze(-1), ori_dst.sin().squeeze(-1)], dim=-1)
    rel_pos = pos_src[local_edge_index[0]] - pos_dst[local_edge_index[1]]

    rel_ori = wrap_angle(ori_src[local_edge_index[0]] - ori_dst[local_edge_index[1]]).squeeze(-1)
    rel_ori_vec = angle_between_2d_vectors(ctr_vector=ori_vec_dst[local_edge_index[1]], nbr_vector=rel_pos)

    if self.attn_cfg.LEARNABLE_PE:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec], dim=-1)
    else:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec, rel_ori_vec], dim=-1)

    emb_model = self.s2s_rel_pe_emb

    rel_pe = emb_model(continuous_inputs=rel_pe_input)

    return rel_pe

  def _scene_fusion(self, batch_map, batch_obs, map_emb, obs_emb, map_mask, obs_mask):

    # N: max number of map tokens
    # A: max number of agents

    from prosim.rollout.distributed_utils import check_mem_usage, print_system_mem_usage, get_gpu_memory_usage

    B, N = map_emb.shape[:2]
    A = obs_emb.shape[1]

    map_batch_idx = torch.arange(B).unsqueeze(1).repeat(1, map_emb.shape[1]).view(-1).to(map_emb.device)[map_mask.view(-1)]
    obs_batch_idx = torch.arange(B).unsqueeze(1).repeat(1, obs_emb.shape[1]).view(-1).to(obs_emb.device)[obs_mask.view(-1)]
    scene_batch_idx = torch.cat([map_batch_idx, obs_batch_idx], dim=0)

    # scene_type: 0 for map tokens, 1 for obs tokens
    scene_type = torch.cat([torch.zeros_like(map_batch_idx), torch.ones_like(obs_batch_idx)], dim=0)

    map_emb_flat = map_emb.view(-1, map_emb.shape[-1])[map_mask.view(-1)]
    obs_emb_flat = obs_emb.view(-1, obs_emb.shape[-1])[obs_mask.view(-1)]
    scene_emb = torch.cat([map_emb_flat, obs_emb_flat], dim=0)

    map_pos = batch_map['position'].view(-1, 2)[map_mask.view(-1)]
    obs_pos = batch_obs['position'].view(-1, 2)[obs_mask.view(-1)]
    scene_pos = torch.cat([map_pos, obs_pos], dim=0)

    map_ori = batch_map['heading'].view(-1, 1)[map_mask.view(-1)]
    obs_ori = batch_obs['heading'].view(-1, 1)[obs_mask.view(-1)]
    scene_ori = torch.cat([map_ori, obs_ori], dim=0)

    obs_edge_index = knn_graph(obs_pos, k=min(self.max_neigh * 4, 100), batch=obs_batch_idx, loop=True)

    scene_edge_index = knn_graph(scene_pos, k=self.max_neigh, batch=scene_batch_idx, loop=True)

    obs_rel_pe = self._get_rel_pe(obs_edge_index, obs_ori, obs_pos, mode='a2a')
    scene_rel_pe = self._get_rel_pe(scene_edge_index, scene_ori, scene_pos, mode='s2s')

    a_mask = scene_type == 1
    x_s = scene_emb

    for i in range(self.num_layers):
      x_s[a_mask] = self.a2a_attn_layers[i](x_s[a_mask], obs_rel_pe, obs_edge_index)
      x_s = self.s2s_attn_layers[i](x_s, scene_rel_pe, scene_edge_index)

    result = {}
    result['obs_mask'] = obs_mask
    result['map_mask'] = map_mask

    result['scene_batch_idx'] = scene_batch_idx
    result['scene_type'] = scene_type
    result['scene_pos'] = scene_pos
    result['scene_ori'] = scene_ori
    result['scene_tokens'] = x_s
    
    result['max_map_num'] = N
    result['max_agent_num'] = A

    return result

  def _update_scene_emb_attn(self, scene_emds):
    # after the observation is updated
    # redo the self-attention of agents and cross-attention from map to agents

    map_mask = scene_emds['scene_type'] == 0
    obs_mask = scene_emds['scene_type'] == 1

    map_pos = scene_emds['scene_pos'][map_mask]
    map_ori = scene_emds['scene_ori'][map_mask]
    obs_pos = scene_emds['scene_pos'][obs_mask]
    obs_ori = scene_emds['scene_ori'][obs_mask]

    map_batch_idx = scene_emds['scene_batch_idx'][map_mask]
    obs_batch_idx = scene_emds['scene_batch_idx'][obs_mask]

    # self-attention of agents
    obs_edge_index = radius_graph(obs_pos, r=self.agent_radius, batch=obs_batch_idx, loop=False, max_num_neighbors=self.max_neigh)

    # cross-attention from map to agent
    a2m_edge_index = radius(x=map_pos, y=obs_pos, r=self.scene_radius, batch_x=map_batch_idx, batch_y=obs_batch_idx, max_num_neighbors=self.max_neigh)
    m2a_edge_index = a2m_edge_index[[1, 0]]

    obs_rel_pe = self._get_rel_pe(obs_edge_index, obs_ori, obs_pos, mode='a2a')
    m2a_pe = self._get_rel_pe_src2dst(m2a_edge_index, obs_ori, obs_pos, map_ori, map_pos)

    x_a = scene_emds['scene_tokens'][obs_mask]
    x_m = scene_emds['scene_tokens'][map_mask]

    for i in range(self.num_layers):
      # self-attention of agents
      x_a = self.a2a_attn_layers[i](x_a, obs_rel_pe, obs_edge_index)

      # self-attention of agents + map
      x_a = self.s2s_attn_layers[i]((x_m, x_a), m2a_pe, m2a_edge_index)
    
    scene_emds['scene_tokens'][obs_mask] = x_a

    return scene_emds

  def _autoregressive_obs_fusion(self, scene_emds, batch_obs, new_obs_emd, new_obs_mask, old_obs_agent_ids):
    B, N, D = new_obs_emd.shape
    
    bidxs, new_oidxs, old_oidxs = [], [], []
    for bidx in range(B):
      old_obs_agent_id = old_obs_agent_ids[bidx]
      new_obs_agent_id = batch_obs['agent_ids'][bidx]
      for new_oidx, agent_id in enumerate(new_obs_agent_id):
        if agent_id in old_obs_agent_id:
          old_oidx = old_obs_agent_id.index(agent_id)
          bidxs.append(bidx)
          new_oidxs.append(new_oidx)
          old_oidxs.append(old_oidx)
    

    new_obs_emd_update = new_obs_emd[bidxs, new_oidxs]
    
    old_obs_mask = scene_emds['obs_mask']
    old_obs_emd = torch.zeros(old_obs_mask.shape[:2] + (D,)).to(new_obs_emd.device)
    old_obs_emd[old_obs_mask] = scene_emds['scene_tokens'][scene_emds['scene_type'] == 1]
    old_obs_emd_update = old_obs_emd[bidxs, old_oidxs]

    new_obs_emd_fuse = self.obs_update_mlp(torch.cat([old_obs_emd_update, new_obs_emd_update], dim=-1))
    
    new_obs_emd[bidxs, new_oidxs] = new_obs_emd_fuse
    
    scene_emds = self._replace_old_obs(scene_emds, batch_obs, new_obs_emd, new_obs_mask)

    return scene_emds

  def _replace_old_obs(self, scene_emds, batch_obs, obs_emb, obs_mask):
    result = {}

    B = obs_emb.shape[0]

    map_token_mask = scene_emds['scene_type'] == 0

    result['map_mask'] = scene_emds['map_mask']

    map_batch_idx = scene_emds['scene_batch_idx'][map_token_mask]
    obs_batch_idx = torch.arange(B).unsqueeze(1).repeat(1, obs_emb.shape[1]).view(-1).to(obs_emb.device)[obs_mask.view(-1)]
    result['scene_batch_idx'] = torch.cat([map_batch_idx, obs_batch_idx], dim=0)

    result['scene_type'] = torch.cat([torch.zeros_like(map_batch_idx), torch.ones_like(obs_batch_idx)], dim=0)

    map_emb_flat = scene_emds['scene_tokens'][map_token_mask]
    obs_emb_flat = obs_emb.view(-1, obs_emb.shape[-1])[obs_mask.view(-1)]
    result['scene_tokens'] = torch.cat([map_emb_flat, obs_emb_flat], dim=0)

    map_pos = scene_emds['scene_pos'][map_token_mask]
    obs_pos = batch_obs['position'].view(-1, 2)[obs_mask.view(-1)]
    result['scene_pos'] = torch.cat([map_pos, obs_pos], dim=0)

    map_ori = scene_emds['scene_ori'][map_token_mask]
    obs_ori = batch_obs['heading'].view(-1, 1)[obs_mask.view(-1)]
    result['scene_ori'] = torch.cat([map_ori, obs_ori], dim=0)

    result['obs_mask'] = obs_mask
    result['max_map_num'] = scene_emds['max_map_num']
    result['max_agent_num'] = obs_emb.shape[1]

    return result

  def update_scene_emb(self, scene_emds, batch_obs, old_obs_agent_ids):
    # scene_emds: [B, N, D]
    # batch_obs: {'input', 'mask'}
    new_obs_emd, new_obs_mask = self.obs_encoder(batch_obs)

    if self.obs_update_cfg.FUSION == 'mlp':
      scene_emds = self._autoregressive_obs_fusion(scene_emds, batch_obs, new_obs_emd, new_obs_mask, old_obs_agent_ids)
    else:
      scene_emds = self._replace_old_obs(scene_emds, batch_obs, new_obs_emd, new_obs_mask)

    if self.obs_update_cfg.ATTN_UPDATE:
      scene_emds = self._update_scene_emb_attn(scene_emds)
    
    return scene_emds