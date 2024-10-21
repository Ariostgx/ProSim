import torch
from torch import nn

from prosim.models.layers.attention_layer import AttentionLayer
from prosim.models.layers.fourier_embedding import FourierEmbeddingFix
from prosim.models.utils.geometry import angle_between_2d_vectors, wrap_angle

from .attn_utils import obtain_valid_edge_node_idex

class ConditionAttn(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.cond_types = self.config.PROMPT.CONDITION.TYPES
    self.cond_cfg = self.config.MODEL.CONDITION_TRANSFORMER
    self.hidden_dim = self.config.MODEL.HIDDEN_DIM
    self.use_cond_type_emd = len(self.cond_types) > 1

    self.use_placeholder = self.cond_cfg.USE_PLACEHOLDER

    self._config_models()

  def _config_models(self):
    if self.use_cond_type_emd:
      self.cond_type_emds = nn.Embedding(len(self.cond_types), self.hidden_dim)

  def _obtain_cond_batch(self, condition_emds):
    '''
    Obtain the condition batch for the condition transformer
    Input:
      condition_emds (dict): the embeddings of the condition agents
        key: condition type
        value: (dict):
          C: number of conditions; C can be different for different condition types
          'emd' (tensor): [B, C, D] - the embedding of the condition agents
          'mask' (tensor): [B, C] - the mask for the condition agents
          'prompt_idx' (tensor): [B, C, 1] - the index of the prompt agent in the scene
    
    Output:
      cond_batch (tensor): [B, C_sum, D] - the condition batch for the condition transformer
      cond_mask_batch (tensor): [B, C_sum] - the mask for the condition batch

      here C_sum is the sum of the number of conditions for all condition types
    '''
    cond_batch = []
    cond_mask_batch = []
    for cond_type in self.cond_types:
      cond_emd_dict = condition_emds[cond_type]
      
      cond_emd = cond_emd_dict['emd'] # [B, C, D]

      if self.use_cond_type_emd:
        cond_idx = torch.tensor(self.cond_types.index(cond_type)).to(cond_emd.device)
        cond_idx = cond_idx.expand_as(cond_emd[:, :, 0]).unsqueeze(-1) # [B, C, 1]
        cond_type_emd = self.cond_type_emds(cond_idx) # [B, C, D]
        cond_emd = cond_emd + cond_type_emd # [B, C, D]
      

      cond_batch.append(cond_emd)
      cond_mask_batch.append(cond_emd_dict['mask'])

    cond_batch = torch.cat(cond_batch, dim=1) # [B, C_sum, D]
    cond_mask_batch = torch.cat(cond_mask_batch, dim=1) # [B, C_sum]

    return cond_batch, cond_mask_batch

  def forward(self, prompt_emd, prompt_mask, condition_emds, **kwargs):
    '''
    Attent the prompt to the conditions
    Input:
      prompt_emd (tensor): [B, N, D] - the embedding of the prompt agents
      prompt_mask (tensor): [B, N] - the mask for the prompt agents
      condition_emds (dict): the embeddings of the condition agents
        key: condition type
        value: (dict):
          C: number of conditions; C can be different for different condition types
          'emd' (tensor): [B, C, D] - the embedding of the condition agents
          'mask' (tensor): [B, C] - the mask for the condition agents
          'prompt_idx' (tensor): [B, C, 1] - the index of the prompt agent in the scene
    Output:
      prompt_condition_emd (tensor): [B, N, D] - the embedding of the prompt agents with condition
    '''

    return prompt_emd


class GNNConditionAttn(ConditionAttn):
  def __init__(self, config):
    super().__init__(config)
    self.pool_type = self.cond_cfg.COND_POOL_FUNC

  def _config_models(self):
    self.rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.hidden_dim/4)
    self.attn_layers = nn.ModuleList(
      [AttentionLayer(hidden_dim=self.hidden_dim, num_heads=self.cond_cfg.NHEAD, head_dim=self.cond_cfg.FF_DIM, dropout=self.cond_cfg.DROPOUT, bipartite=False, has_pos_emb=True) for _ in range(self.cond_cfg.NLAYER)]
      )

  def _get_rel_pe(self, local_edge_index, ori_dst, pos_dst, ori_src, pos_src):
    '''
    Obtain the relative positional encoding for the GNN based on the relative position and orientation
    '''

    ori_vec_dst = torch.stack([ori_dst.cos().squeeze(-1), ori_dst.sin().squeeze(-1)], dim=-1)
    rel_pos = pos_src[local_edge_index[0]] - pos_dst[local_edge_index[1]]

    rel_ori = wrap_angle(ori_src[local_edge_index[0]] - ori_dst[local_edge_index[1]]).squeeze(-1)
    rel_ori_vec = angle_between_2d_vectors(ctr_vector=ori_vec_dst[local_edge_index[1]], nbr_vector=rel_pos)
    rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec, rel_ori_vec], dim=-1)

    rel_pe = self.rel_pe_emb(continuous_inputs=rel_pe_input)

    return rel_pe

  def _construct_cond_edge_matrix(self, prompt_mask, condition_emds):
    '''
    Construct conditional edge matrix for the GNN
    Input:
      prompt_mask: [B, N] - the mask for the prompt agents
      condition_emds (dict): the embeddings of the condition agents
        key: condition type
        value: (dict):
          C: number of conditions; C can be different for different condition types
          'emd' (tensor): [B, C, D/2D] - the embedding of the condition agents. D for the source-source edge, 2D for the source-target and target-source edge (if bipartite)
          'mask' (tensor): [B, C] - the mask for the condition agents
          'prompt_idx' (tensor): [B, C, 1/2] - the index of the prompt agent in the scene. 1 for the source agent, 2 for the target agent (if bipartite)
    
    Output:
      edge_attr (tensor): [B, N, N, M, D] - the edge attribute matrix, where M is the number of condition types
      edge_mask (tensor): [B, N, N, M] - the mask for the edge attribute matrix
    '''
    B, N = prompt_mask.shape
    device = prompt_mask.device
    M = len(condition_emds)

    edge_attr = torch.zeros(B, N, N, M, self.hidden_dim, device=device)
    edge_mask = torch.zeros(B, N, N, M, device=device, dtype=torch.bool)

    for cond_idx, cond_type in enumerate(list(condition_emds.keys())):
      cond_emd_dict = condition_emds[cond_type]
      cond_emd = cond_emd_dict['emd'] # [B, C, D/2D]
      cond_mask = cond_emd_dict['mask'] # [B, C]
      prompt_idx = cond_emd_dict['prompt_idx'] # [B, C, 1/2]

      C = cond_emd.shape[1]
      
      s_emd = cond_emd[..., :self.hidden_dim][cond_mask] # [V, D] where V is the number of valid conditions
      bidx = torch.arange(B, device=device).unsqueeze(-1).expand(B, C)[cond_mask] # [V]
      s_nidx = prompt_idx[..., 0][cond_mask] # [V]
      m_idx = torch.ones_like(s_nidx, device=device) * cond_idx # [V]
      
      is_bipartite = cond_emd.shape[-1] > self.hidden_dim
      if is_bipartite:
        t_emd = cond_emd[..., self.hidden_dim:][cond_mask] # [V, D]
        t_nidx = prompt_idx[..., 1][cond_mask] # [V]

        edge_attr[bidx, s_nidx, t_nidx, m_idx] = s_emd
        edge_attr[bidx, t_nidx, s_nidx, m_idx] = t_emd
        
        edge_mask[bidx, s_nidx, t_nidx, m_idx] = True
        edge_mask[bidx, t_nidx, s_nidx, m_idx] = True
      
      else:
        edge_attr[bidx, s_nidx, s_nidx, m_idx] = s_emd
        edge_mask[bidx, s_nidx, s_nidx, m_idx] = True
  
    return edge_attr, edge_mask

  def _pool_edges(self, edge_attr, edge_mask):
    '''
    Pool the edge attribute matrix
    Input:
      edge_attr: [B, N, N, M, D] - the edge attribute matrix
      edge_mask: [B, N, N, M] - the mask for the edge attribute matrix
    Output:
      edge_attr: [B, N, N, D] - the pooled edge attribute matrix
      edge_mask: [B, N, N] - the mask for the pooled edge attribute matrix
    '''

    if self.pool_type == 'mean':
      edge_attr = edge_attr.sum(dim=-2) # [B, N, N, D]
      edge_cnt = edge_mask.sum(dim=-1).clamp(min=1)[..., None] # [B, N, N, 1] - avoid division by zero
      edge_attr = edge_attr / edge_cnt
    
    elif self.pool_type == 'max':
      edge_attr[~edge_mask] = torch.tensor(float('-inf'), device=edge_attr.device)
      edge_attr, _ = edge_attr.max(dim=-2) # [B, N, N, D]
    
    return edge_attr, edge_mask.any(dim=-1) # [B, N, N, D], [B, N, N]

  def forward(self, prompt_emd, prompt_mask, condition_emds, **kwargs):
    '''
      E: number of valid edges
      P: number of valid nodes
    '''
    # obtain prompt position and heading for the edge attribute matrix
    node_mask = prompt_mask # [B, N], node_mask.sum() == P

    prompt_pos = kwargs['position'][node_mask] # [P, 2]
    prompt_ori = kwargs['heading'][node_mask] # [P, 1]

    if len(condition_emds) == 0:
      return prompt_emd
    
    edge_attr, edge_mask = self._construct_cond_edge_matrix(prompt_mask, condition_emds) # [B, N, N, M, D], [B, N, N, M]
    edge_attr, edge_mask = self._pool_edges(edge_attr, edge_mask) # [B, N, D], [B, N]

    edge_node_index = obtain_valid_edge_node_idex(edge_mask, node_mask) # [E, 2] - range [0, P)

    E = edge_mask.sum() # number of valid edges
    
    # get the edge attribute for the valid edges
    edge_attr = edge_attr[edge_mask] # [E, D]

    # get the relative positional encoding for the valid edges
    rel_pe = self._get_rel_pe(edge_node_index, prompt_ori, prompt_pos, prompt_ori, prompt_pos) # [E, D]

    # add the relative positional encoding to the edge attribute
    edge_attr = edge_attr + rel_pe

    x_p = prompt_emd[node_mask] # [P, D]

    # pass through the attention layers
    for attn_layer in self.attn_layers:
      x_p = attn_layer(x_p, edge_attr, edge_node_index)
    
    prompt_emd[node_mask] += x_p

    return prompt_emd

condition_attns = {'gnn': GNNConditionAttn}