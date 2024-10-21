import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, knn

from prosim.models.layers.mlp import MLP, CG_stacked
from prosim.models.utils.pos_enc import pos2posemb
from prosim.models.layers.attention_layer import AttentionLayer
from prosim.models.layers.fourier_embedding import FourierEmbedding, FourierEmbeddingFix
from prosim.models.utils.geometry import angle_between_2d_vectors, wrap_angle

class ActDecoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.m_cfg = config.MODEL.POLICY.ACT_DECODER
    self.c_cfg = self.m_cfg.CONTEXT
    self.r_cfg = config.ROLLOUT.POLICY

    self.k_pred_mode = self.m_cfg.TRAJ.PRED_MODE
    self.motion_K = self.m_cfg.TRAJ.K

    self.state_dim = len(config.DATASET.FORMAT.TARGET.ELEMENTS.split(','))

    if self.m_cfg.TRAJ.PRED_GMM:
      self.state_dim += 3 # std1, std2, rho

    self.step = config.DATASET.FORMAT.TARGET.STEPS
    self.output_dim = self.state_dim * self.step
    self.d_dim = self.config.MODEL.HIDDEN_DIM
    self.use_ped_cycl = self.config.DATASET.USE_PED_CYCLIST

    self._config_models()
  
  def _config_context(self):
    if self.c_cfg.GOAL or self.c_cfg.GT_GOAL:
      if self.c_cfg.USE_POSE_EMB:
        self.goal_encoder = MLP([self.d_dim, self.d_dim])
      else:
        self.goal_encoder = MLP([2, self.d_dim])

      if self.c_cfg.EMD:
        self.context_fuse = MLP([self.d_dim * 2, self.d_dim])

  def _config_fusion(self):
    layer = self.m_cfg.MCG.LAYER
    self.CG_fuse = CG_stacked(layer, self.d_dim)

  def _config_pred(self):
    if self.k_pred_mode == 'vel_pred':
      self.vel_head = MLP([self.d_dim, self.d_dim, self.d_dim//2, 2], ret_before_act=True)
      return

    if self.k_pred_mode == 'goal_pred':
      self.goal_head = MLP([self.d_dim, 3], ret_before_act=True)
      return

    if self.k_pred_mode == 'mlp':
      self.motion_head = MLP([self.d_dim, self.d_dim, self.d_dim//2, self.motion_K * self.output_dim], ret_before_act=True)
    else:
      self.motion_head = MLP([self.d_dim, self.d_dim, self.d_dim//2, self.output_dim], ret_before_act=True)
      self.CG_decode = CG_stacked(3, self.d_dim)

    if self.k_pred_mode == 'anchor':
      num_types = 3 if self.use_ped_cycl else 1
      self.motion_anchors = nn.Embedding(self.motion_K * num_types, self.d_dim)
    
    elif self.k_pred_mode == 'cluster':
      file = self.m_cfg.TRAJ.CLUSTER_PATH
      k_goals = torch.from_numpy(np.load(file)).float()
      self.k_goals_pe = FourierEmbeddingFix(num_pos_feats=self.d_dim//2)(k_goals)
      self.cluster_mlp = MLP([self.d_dim, self.d_dim])

    if self.config.LOSS.ROLLOUT_TRAJ.USE_GOAL_PRED_LOSS:
      self.pred_mlp = MLP([self.d_dim, self.d_dim, self.d_dim//2, 2], ret_before_act=True)
      
  def _compute_traj(self, pred_feat, policy_emd):
    b = pred_feat.shape[0]
    device = pred_feat.device
    
    if self.k_pred_mode == 'vel_pred':
      vel_pred = self.vel_head(pred_feat)
      return {'init_vel_pred': vel_pred}
  
    if self.k_pred_mode == 'goal_pred':
      goal_pred = self.goal_head(pred_feat)
      return {'goal_pred': goal_pred}

    if self.k_pred_mode == 'mlp':
      motion = self.motion_head(pred_feat).view(b, self.motion_K, self.step, self.state_dim)
    
    else:
      if self.k_pred_mode == 'anchor':
        type_idx = (policy_emd['agent_type'] - 1) * self.motion_K
        type_idx = type_idx.unsqueeze(-1).repeat(1, self.motion_K)

        # anchor index for each motion mode for each agent type
        anchor_index = torch.arange(self.motion_K)[None, :].repeat(b, 1).to(device)
        anchor_index = anchor_index + type_idx

        anchor_emd = self.motion_anchors(anchor_index)
      
      elif self.k_pred_mode == 'cluster':
        anchor_emd = self.cluster_mlp(self.k_goals_pe.to(device))
        anchor_emd = anchor_emd[None, :].repeat(b, 1, 1)

      mask = torch.ones([b, self.motion_K]).to(torch.bool).to(device)
      pred_emd , _ = self.CG_decode(anchor_emd, pred_feat, mask)
      
      motion = self.motion_head(pred_emd).view(b, self.motion_K, self.step, self.state_dim)

    if self.m_cfg.RANDOM_NOISE_STD > 0:
      motion[..., :2] += torch.randn_like(motion[..., :2]) * self.m_cfg.RANDOM_NOISE_STD
      print('WARNING: add random noise to predicted action with std: ', self.m_cfg.RANDOM_NOISE_STD)

    traj_pred = motion[..., :2].cumsum(dim=-2)
    head_pred = wrap_angle(motion[..., 2:3].cumsum(dim=-2))
    gmm_param = motion[..., 3:]

    motion_pred = torch.cat([traj_pred, head_pred, gmm_param], dim=-1)
    
    # we do not need to predict the goal for the agent if we are doing online rollout
    motion_prob = torch.ones_like(motion[..., 0, 0], device=device)

    result = {'motion_pred': motion_pred, 'motion_prob': motion_prob}

    if self.config.LOSS.ROLLOUT_TRAJ.USE_GOAL_PRED_LOSS:
      reconst_pred = self.pred_mlp(policy_emd['emd'])
      result['reconst_pred'] = reconst_pred
    
    if 'prompt_loss' in policy_emd:
      result['prompt_loss'] = policy_emd['prompt_loss']

    return result 
  
  def _config_models(self):
    self._config_context()
    self._config_fusion()
    self._config_pred()

  def _extract_context(self, policy_emd):
    context = []
    c_cfg = self.c_cfg
    
    # do not use pred goal and gt goal at the same time
    if c_cfg.GOAL:
      if c_cfg.GT_GOAL and 'gt_goal' in policy_emd:
        goal_cond = policy_emd['gt_goal']
      else:
        goal_cond = policy_emd['goal']

      if self.c_cfg.USE_POSE_EMB:
        goal_cond = pos2posemb(goal_cond, self.d_dim // 2)

      context.append(self.goal_encoder(goal_cond))

    if self.c_cfg.EMD:
      context.append(policy_emd['emd'])

    if len(context) > 1:
      context = self.context_fuse(torch.cat(context, dim=-1))
      return context
  
    return context[0]
      
  def forward(self, policy_emd, scene_emd):
    scene_tokens = scene_emd['scene_tokens']
    scene_mask = scene_emd['scene_mask']

    context_emd = self._extract_context(policy_emd)

    _, fuse_feature = self.CG_fuse(scene_tokens, context_emd, scene_mask)
    
    return fuse_feature

class AttnRelPE(ActDecoder):
  def _config_fusion(self):
    self.attn_cfg = self.config.MODEL.POLICY.ACT_DECODER.ATTN
    
    if self.attn_cfg.LEARNABLE_PE:
      self.a2p_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.d_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
      self.m2p_rel_pe_emb = FourierEmbedding(input_dim=3, hidden_dim=self.d_dim, num_freq_bands=self.attn_cfg.PE_NUM_FREQ)
    else:
      self.a2p_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.d_dim/4)
      self.m2p_rel_pe_emb = FourierEmbeddingFix(num_pos_feats=self.d_dim/4)

    # agent to policy
    self.a2p_attn_layers = nn.ModuleList(
      [AttentionLayer(hidden_dim=self.d_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=True, has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
      )
  
    # map to policy
    self.m2p_attn_layers =  nn.ModuleList(
      [AttentionLayer(hidden_dim=self.d_dim, num_heads=self.attn_cfg.NUM_HEAD, head_dim=self.attn_cfg.FF_DIM, dropout=self.attn_cfg.DROPOUT,bipartite=True,has_pos_emb=True) for _ in range(self.attn_cfg.NUM_LAYER)]
      )
  
    self.num_layers = self.attn_cfg.NUM_LAYER
    self.agent_radius = self.attn_cfg.AGENT_RADIUS
    self.map_radius = self.attn_cfg.MAP_RADIUS
    self.max_neigh = self.attn_cfg.MAX_NUM_NEIGH

  def _get_rel_pe(self, local_edge_index, ori_dst, pos_dst, ori_src, pos_src, mode='a2p'):

    ori_vec_dst = torch.stack([ori_dst.cos().squeeze(-1), ori_dst.sin().squeeze(-1)], dim=-1)
    
    rel_pos = pos_src[local_edge_index[0]] - pos_dst[local_edge_index[1]]

    rel_ori = wrap_angle(ori_src[local_edge_index[0]] - ori_dst[local_edge_index[1]]).squeeze(-1)
    rel_ori_vec = angle_between_2d_vectors(ctr_vector=ori_vec_dst[local_edge_index[1]], nbr_vector=rel_pos)

    if self.attn_cfg.LEARNABLE_PE:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec], dim=-1)
    else:
      rel_pe_input = torch.stack([torch.norm(rel_pos, dim=-1), rel_ori, rel_ori_vec, rel_ori_vec], dim=-1)

    emb_model = self.a2p_rel_pe_emb if mode == 'a2p' else self.m2p_rel_pe_emb

    rel_pe = emb_model(continuous_inputs=rel_pe_input)

    return rel_pe, rel_pe_input


  def _process_scene_token(self, batch_scene):
    scene_emd = batch_scene['input']
    scene_mask = batch_scene['mask']

    assert scene_emd.shape[:2] == scene_mask.shape[:2]

    B = scene_emd.shape[0]
    emd_flat = scene_emd.view(-1, scene_emd.shape[-1])[scene_mask.view(-1)]
    pos = batch_scene['pos'].view(-1, 2)[scene_mask.view(-1)]
    ori = batch_scene['ori'].view(-1, 1)[scene_mask.view(-1)]

    batch_idx = torch.arange(B).unsqueeze(1).repeat(1, scene_emd.shape[1]).view(-1).to(scene_emd.device)[scene_mask.view(-1)]

    return emd_flat, pos, ori, batch_idx

  def attn_fuse(self, policy_emd, policy_batch_idx, batch_obs, batch_map, batch_pos):
    from prosim.rollout.distributed_utils import check_mem_usage, print_system_mem_usage, get_gpu_memory_usage

    B = policy_emd.shape[0]
    policy_pos = batch_pos['position']
    policy_ori = batch_pos['heading']

    obs_emd, obs_pos, obs_ori, obs_batch_idx = self._process_scene_token(batch_obs)
    map_emd, map_pos, map_ori, map_batch_idx = self._process_scene_token(batch_map)
    
    if self.config.MODEL.REL_POS_EDGE_FUNC == 'radius':
      p2a_edge_index = radius(x=obs_pos, y=policy_pos, r=self.agent_radius, batch_x=obs_batch_idx, batch_y=policy_batch_idx, max_num_neighbors=self.max_neigh)
    else:
      p2a_edge_index = knn(x=obs_pos, y=policy_pos, k=self.max_neigh, batch_x=obs_batch_idx, batch_y=policy_batch_idx)
      
    a2p_edge_index = p2a_edge_index[[1, 0]]
    
    a2p_pe, a2p_pe_input = self._get_rel_pe(a2p_edge_index, policy_ori, policy_pos, obs_ori, obs_pos, mode='a2p')

    if self.config.MODEL.REL_POS_EDGE_FUNC == 'radius':
      p2m_edge_index = radius(x=map_pos, y=policy_pos, r=self.map_radius, batch_x=map_batch_idx, batch_y=policy_batch_idx, max_num_neighbors=self.max_neigh)
    else:
      p2m_edge_index = knn(x=map_pos, y=policy_pos, k=self.max_neigh, batch_x=map_batch_idx, batch_y=policy_batch_idx)
    m2p_edge_index = p2m_edge_index[[1, 0]]
      
    m2p_pe, m2p_pe_input = self._get_rel_pe(m2p_edge_index, policy_ori, policy_pos, map_ori, map_pos, mode='m2p')

    x_p = policy_emd
    x_a = obs_emd
    x_m = map_emd

    for i in range(self.num_layers):
      x_p = self.a2p_attn_layers[i]((x_a, x_p), a2p_pe, a2p_edge_index)
      x_p_updated = self.m2p_attn_layers[i]((x_m, x_p), m2p_pe, m2p_edge_index)
      if self.attn_cfg.NOT_USE_MAP:
        x_p = x_p + x_p_updated * 0.0
        print('WARNING: not using map in policy attention')
      else:
        x_p = x_p_updated

    return x_p

  def forward(self, policy_emd, batch_obs, batch_map, batch_pos):
    context_emd = self._extract_context(policy_emd)

    fuse_feature = self.attn_fuse(context_emd, batch_obs, batch_map, batch_pos)

    result = self._compute_traj(fuse_feature, policy_emd)

    return result