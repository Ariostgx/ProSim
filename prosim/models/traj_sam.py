import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from prosim.core.registry import registry
from prosim.models.utils.visualization import vis_agent_traj_pred, vis_scene_traj_pred
from prosim.models.utils.geometry import wrap_angle, batch_rotate_2D, rel_traj_coord_to_last_step, rel_vel_coord_to_last_step
from prosim.models.condition_transformer import ConditionTransformer

from .base import BaseModel

@registry.register_model(name='prosim_policy_relpe_T_step_temporal_close_loop')
class ProSim(BaseModel):
  def __init__(self, config):
    super().__init__(config)
    self._config_models()
    self._config_parameters()

    self.rollout_steps = config.ROLLOUT.POLICY.REPLAN_FREQ
    self.rollout_top_k = config.ROLLOUT.POLICY.TOP_K
    self.rollout_top_k_train = config.ROLLOUT.POLICY.TOP_K_TRAIN
    self.hist_step = config.DATASET.FORMAT.HISTORY.STEPS

    self.pred_gmm = self.config.MODEL.POLICY.ACT_DECODER.TRAJ.PRED_GMM
    self.pred_vel = self.config.MODEL.POLICY.ACT_DECODER.TRAJ.PRED_VEL
  
  def _config_models(self):
    model_cfg = self.config.MODEL
    self.scene_encoder = registry.get_scene_encoder(model_cfg.SCENE_ENCODER.TYPE)(self.config, model_cfg.SCENE_ENCODER)

    self.prompt_encoder = nn.ModuleDict()

    self.use_condition = len(self.config.PROMPT.CONDITION.TYPES) > 0

    for task in self.tasks:
      encoder_name = self.config.TASK[task.upper()].PROMPT
      self.prompt_encoder[task] = registry.get_prompt_encoder(encoder_name)(self.config)
    
    self.decoder = registry.get_decoder(model_cfg.DECODER.TYPE)(model_cfg)

    self.models = [self.scene_encoder, self.decoder]
    for task in self.tasks:
      self.models.append(self.prompt_encoder[task])
    
    if self.use_condition:
      self.condition_locations = model_cfg.CONDITION_TRANSFORMER.CONDITION_LOCATIONS
      
      self.condition_transformers = nn.ModuleDict()
      for location in self.condition_locations:
        self.condition_transformers[location] = ConditionTransformer(self.config)
        self.models.append(self.condition_transformers[location])

    model_cfg = self.config.MODEL

    self.policy = registry.get_policy(model_cfg.POLICY.TYPE)(self.config)
    self.models += [self.policy]

  def forward(self, batch, mode):
    '''
    Forward pass for the model. This is the main function of ProSim.
    1. Obtain initial scene embedding from the scene encoder.
    2. Encode the prompt for all tasks.
    3. Decode the batch with closed-loop rollout.
    '''
    self.mode = mode # train or eval
    scene_embs = self.encode_scene(batch)
    prompt_encs = self.encode_prompt(batch)
    output = self.decode_batch(scene_embs, prompt_encs, batch, mode)

    return output
  
  def encode_scene(self, batch):
    '''
    Encode the initial observation and map with the scene encoder.
    '''
    return self.scene_encoder(batch.extras['init_obs'], batch.extras['init_map'])

  def encode_prompt(self, batch, prompt_dict={}):
    '''
    Encode the prompt for all tasks.
    '''
    prompt_encs = {}
    all_tasks = self.tasks if len(prompt_dict) == 0 else prompt_dict.keys()
    
    for task in all_tasks:
      if task in prompt_dict:
        prompt_data = prompt_dict[task]
      else:
        prompt_data = batch.extras['prompt'][task]

      prompt_emd, _ = self.prompt_encoder[task](prompt_data)

      if self.use_condition and 'prompt_encoder' in self.condition_locations:
        prompt_emd = self.condition_transformers['prompt_encoder'](prompt_emd, prompt_data['prompt_mask'], batch.extras['condition'])
      
      prompt_data['prompt_emd'] = prompt_emd

      prompt_encs[task] = prompt_data
    
    return prompt_encs

  def decode_batch(self, scene_embs, prompt_encs, batch, mode):
    '''
    Decode the batch with closed-loop rollout.
    1. decode the policy for all agents with scene embedding and prompt
    2. initialize the agent trajectories with history observations
    3. rollout the trajectories with the decoded policy
    '''
    policy_emds = self.generate_policy(batch, scene_embs, prompt_encs)
    policy_agent_ids = {task: batch.extras['prompt'][task]['agent_ids'] for task in self.tasks}

    all_t_indices = sorted(batch.extras['all_t_indices'].cpu().numpy().tolist())
    agent_trajs = self.init_agent_trajs(policy_agent_ids, batch)

    return self.rollout_batch(batch, scene_embs, policy_emds, policy_agent_ids, agent_trajs, all_t_indices, mode)
  
  def generate_policy(self, batch, scene_embs, prompt_encs):
    '''
    Generate the policy embedding for all agents.
    1. obtain policy embedding with each agent's prompt (initial status, type and size).
    2. apply condition transformer on the policy embedding if specified.
    '''
    policy_emds = {}
    for task, prompt_enc in prompt_encs.items():
      # get policy embedding from decoder (decode each agent's prompt to a policy embedding)
      policy_emds[task] = self.decoder(scene_embs, prompt_enc)

      if self.use_condition and 'policy_decoder' in self.condition_locations:
        policy_emd = policy_emds[task]['emd']
        policy_mask = prompt_enc['prompt_mask']
        policy_position = prompt_enc['position']
        policy_heading = prompt_enc['heading']

        policy_emd, prompt_loss = self.condition_transformers['policy_decoder'](batch.extras['condition'], position=policy_position, heading=policy_heading, scene_embs=scene_embs, prompt_emd=policy_emd, prompt_mask=policy_mask, batch=batch)
        
        policy_emds[task]['emd'] = policy_emd

        if prompt_loss is not None:
          policy_emds[task]['prompt_loss'] = prompt_loss
    
    return policy_emds

  def rollout_batch(self, batch, scene_embs, policy_emds, policy_agent_ids, agent_trajs, all_t_indices, mode):
    '''
    Conduct closed-loop rollout for all agents in the batch.
    1. update scene embedding with the predicted future observations
    2. decode the policy output with the updated scene embedding
    3. update the agent trajectories with the decoded policy output
    4. repeat the process until the end of the rollout horizon
    '''

    model_outputs = []

    # if policy_emds has k predictions, select one to rollout
    tasks = list(policy_emds.keys())
    policy_emds[tasks[0]] = self._select_k_emd_from_batch(policy_emds[tasks[0]], batch)

    for t in all_t_indices:
      scene_embs, agent_positions = self.step_env(scene_embs, agent_trajs, batch, policy_agent_ids, t, all_t_indices)

      if t == 0:
        latent_state_dict = None
      else:
        latent_state_dict = {}
        latent_state_dict['state'] = model_outputs[-1][self.tasks[0]]['latent_state']
        latent_state_dict['agent_name'] = ['-'.join(name.split('-')[:2]) for name in model_outputs[-1][self.tasks[0]]['pair_names']]
      
      model_output = self.decode_output(policy_emds, scene_embs, policy_agent_ids, batch, agent_positions, t, latent_state_dict)
      agent_trajs = self.step_agent_traj(agent_trajs, model_output, policy_agent_ids, t, mode)

      model_outputs.append(model_output)

    result = self._process_rollout(agent_trajs, model_outputs, policy_agent_ids)
    return result 
  

  def decode_output(self, policy_emds, scene_embs, policy_agent_ids, batch, agent_positions=None, target_t=None, latent_state_dict=None):
    '''
    Given the policy embedding and scene embedding, decode the output for all agents in the batch.
    1. organize all agents across batch and task dimensions into a single batch input; this is to ensure the batch size is the full number of i/o pairs considered to optimize memory usage.
    2. obtain the policy output for all agents
    '''
    model_outputs = {}
    tasks = list(policy_emds.keys())

    task = tasks[0]
    batch_policy_emd, batch_obs, batch_map, batch_pos, batch_pair_names = \
      self._get_policy_batch_input(batch, policy_agent_ids[task], policy_emds[task], scene_embs, agent_positions, target_t)

    if latent_state_dict is None:
      latent_state = None
    else:
      latent_state = self.policy.format_latent_state(latent_state_dict, [batch_pair_names])

    all_output = self.get_action(batch_policy_emd, batch_obs, batch_map, batch_pos, [batch_pair_names], latent_state=latent_state)

    model_outputs = {}
    model_outputs[task] = all_output
    model_outputs[task]['pair_names'] = batch_pair_names

    return model_outputs


  def step_env(self, scene_embs, a_traj, batch, policy_agent_ids, t, all_t_indices):
    '''
    Update the scene embedding with the predicted future observations.
    '''
    task = 'motion_pred'

    a_pos = {}
    tidx = a_traj[task]['last_step']
    a_pos['position'] = a_traj[task]['init_pos'] + a_traj[task]['traj'][..., tidx-1, :2]
    a_theta = torch.arctan2(a_traj[task]['traj'][..., tidx-1, 2], a_traj[task]['traj'][..., tidx-1, 3])
    a_pos['heading'] = wrap_angle(a_theta[:, :, None] + a_traj[task]['init_heading'])

    t_idx = all_t_indices.index(t)

    if t_idx == 0:
      scene_embs_next = scene_embs
    else:
      # fut_obs contains observation for all agents at timestep t
      fut_obs = batch.extras['fut_obs'][t]

      if t_idx > 1:
        old_obs = batch.extras['fut_obs'][all_t_indices[t_idx-1]]
      else:
        old_obs = batch.extras['init_obs']
      old_obs_agent_ids = old_obs['agent_ids']

      # to support M rollout, repeat the old_obs_agent_ids for init obs
      B = len(policy_agent_ids[task])
      if len(old_obs_agent_ids) == 1 and B > 1:
        old_obs_agent_ids = old_obs_agent_ids * B

      task_policy_agent_ids = policy_agent_ids[task]

      bidxs = []
      oidxs = []
      nidxs = []
      rel_trajs = []
      positions = []
      headings = []

      for bidx in range(len(task_policy_agent_ids)):
        for nidx, agent_id in enumerate(task_policy_agent_ids[bidx]):
          bidxs.append(bidx)
          oidxs.append(fut_obs['agent_ids'][bidx].index(agent_id))
          nidxs.append(nidx)

      # obtain self.hist_step+2 past observations (first two steps only used to compute vel_acc)
      abs_trajs = a_traj[task]['traj'][bidxs, nidxs, tidx-self.hist_step-2:tidx]
      positions = a_pos['position'][bidxs, nidxs]
      headings = a_pos['heading'][bidxs, nidxs]

      rel_trajs = rel_traj_coord_to_last_step(abs_trajs)

      if self.pred_vel:
        rel_vel = a_traj[task]['vel'][bidxs, nidxs, tidx-self.hist_step-1:tidx]
        rel_vel = rel_vel_coord_to_last_step(abs_trajs, rel_vel)
        rel_vel_acc = self._get_rel_vel_acc(rel_trajs[..., :2], rel_vel)
      else:
        rel_vel_acc = self._get_rel_vel_acc(rel_trajs[..., :2])

      # update observation based on timestep
      fut_obs['input'][bidxs, oidxs, :self.hist_step, :4] = rel_trajs[:, -self.hist_step:]
      fut_obs['input'][bidxs, oidxs, :self.hist_step, 4:8] = rel_vel_acc
      fut_obs['position'][bidxs, oidxs] = positions
      fut_obs['heading'][bidxs, oidxs] = headings.squeeze(-1)
      fut_obs['mask'][bidxs, oidxs, :self.hist_step] = True

      scene_embs_next = self._update_scene_emb(scene_embs, fut_obs, old_obs_agent_ids)
    
    return scene_embs_next, a_pos

  def step_agent_traj(self, a_traj, model_output, policy_agent_ids, t, mode):
    '''
    Update the agent trajectories with the predicted future observations.
    '''
    task = self.tasks[0]

    bidxs = []
    nidxs = []
    pidxs = []

    curr_trajs = []
    pred_trajs = []

    for bidx, agent_ids in enumerate(policy_agent_ids[task]):
      for nidx, agent_id in enumerate(agent_ids):
        pred_name = f'{bidx}-{agent_id}-{t}'
        if pred_name not in model_output[task]['pair_names']:
          continue
        pidx = model_output[task]['pair_names'].index(pred_name)
        
        bidxs.append(bidx)
        nidxs.append(nidx)
        pidxs.append(pidx)
    
    P = len(pidxs)
    all_probs = model_output[task]['motion_prob'][pidxs]

    if mode == 'train':
      rollout_k = self.rollout_top_k_train
    else:
      rollout_k = self.rollout_top_k
    
    # make sure rollout_k is less than or equal to the number of predictions
    rollout_k = min(rollout_k, all_probs.shape[1])

    _, traj_idxs_top_k = torch.topk(all_probs, rollout_k, dim=1)
    rand_idxs = torch.randint(0, rollout_k, (P,), device=self.device)
    traj_idxs = traj_idxs_top_k[torch.arange(P, device=self.device), rand_idxs]

    tidx = a_traj[task]['last_step']
    curr_trajs = a_traj[task]['traj'][bidxs, nidxs, :tidx]
    pred_trajs = model_output[task]['motion_pred'][pidxs, traj_idxs, :self.rollout_steps]

    if not self.config.MODEL.BPTT:
      pred_trajs = pred_trajs.detach()

    last_thetas = torch.arctan2(curr_trajs[:, -1, 2], curr_trajs[:, -1, 3])[:, None]
    pred_xys = batch_rotate_2D(pred_trajs[:, :, :2], last_thetas) + curr_trajs[:, -1:, :2]
    pred_thetas = wrap_angle(last_thetas + pred_trajs[:, :, 2])

    fut_trajs = torch.cat([pred_xys, torch.sin(pred_thetas)[..., None], torch.cos(pred_thetas)[..., None]], dim=-1)

    curr_step = tidx + self.rollout_steps
    
    B, N = a_traj[task]['traj'].shape[:2]
    new_traj = torch.zeros(B, N, self.rollout_steps, 4, device=self.device)
    new_traj[bidxs, nidxs] = fut_trajs
    a_traj[task]['traj'] = torch.cat([a_traj[task]['traj'], new_traj], dim=2)
    a_traj[task]['last_step'] = curr_step


    if self.pred_vel:
      if self.pred_gmm:
        pred_vel = pred_trajs[..., 6:8]
      else:
        pred_vel = pred_trajs[..., 3:5]
      pred_vel = batch_rotate_2D(pred_vel, last_thetas)

      new_vel = torch.zeros(B, N, self.rollout_steps, 2, device=self.device)
      new_vel[bidxs, nidxs] = pred_vel

      a_traj[task]['vel'] = torch.cat([a_traj[task]['vel'], new_vel], dim=2)
      
    return a_traj

  def _get_io_position_from_dict(self, agent_positions, pair_name):
    a_name = '-'.join(pair_name.split('-')[:2])
    return {'position': agent_positions[a_name]['position'], 'heading': agent_positions[a_name]['heading'][..., None]}


  def _scene_emd_to_batch(self, scene_embs_T, mode='obs'):
    '''
    Convert the scene embedding to a batch input. 
    Organize the scene embedding for all timesteps and all agents into a single batch input.
    '''
    T = len(scene_embs_T)
    B = scene_embs_T[0]['obs_mask'].shape[0]
    s_name = 'max_agent_num' if mode == 'obs' else 'max_map_num'
    S = max([emds[s_name] for emds in scene_embs_T])

    D = scene_embs_T[0]['scene_tokens'].shape[-1]
    dtype = scene_embs_T[0]['scene_tokens'].dtype

    batch_input  = torch.zeros(B, T, S, D, device=self.device, dtype=dtype)
    batch_mask = torch.zeros(B, T, S, device=self.device).to(torch.bool)
    batch_pos = torch.zeros(B, T, S, 2, device=self.device)
    batch_ori = torch.zeros(B, T, S, 1, device=self.device)

    for t in range(T):
      scene_embs = scene_embs_T[t]
      # map - 0, obs - 1
      type_id = 1 if mode == 'obs' else 0
      type_mask = scene_embs['scene_type'] == type_id
      scene_batch_idx = scene_embs['scene_batch_idx'][type_mask]

      mask_name = 'obs_mask' if mode == 'obs' else 'map_mask'
      scene_mask = scene_embs[mask_name]
      scene_mask = torch.cat([scene_mask, torch.zeros(B, S - scene_mask.shape[1], device=self.device).to(torch.bool)], dim=1)
      
      batch_mask[:, t, :] = scene_mask

      B_idxs = scene_batch_idx
      S_idxs = torch.arange(S, device=self.device)[None, :].repeat(B, 1)[scene_mask]

      batch_input[B_idxs, t, S_idxs] = scene_embs['scene_tokens'][type_mask]
      batch_pos[B_idxs, t, S_idxs] = scene_embs['scene_pos'][type_mask]
      batch_ori[B_idxs, t, S_idxs] = scene_embs['scene_ori'][type_mask]
    
    # convert to [BT, S, D]
    batch_input = batch_input.view(B*T, S, D)
    batch_mask = batch_mask.view(B*T, S)
    batch_pos = batch_pos.view(B*T, S, 2)
    batch_ori = batch_ori.view(B*T, S, 1)
    
    return {'input': batch_input, 'mask': batch_mask, 'pos': batch_pos, 'ori': batch_ori}

  def _select_k_emd_from_batch(self, policy_emds, batch):
    '''
    Select one of the k policy embeddings for each agent in the batch

    Selection critiria:
      Training:
        select the policy embedding that has the closest goal point to the ground truth goal
      Inference:
        select the policy embedding that from the top-k goal probability
    '''

    # does not need selection
    if policy_emds['emd'].ndim == 3:
      return policy_emds
    
    goal_prob = policy_emds['goal_prob'] # B, N, K

    goal_point = policy_emds['goal_point'] # B, N, 2

    B, N, K, D = policy_emds['emd'].shape
    if self.mode == 'train':
      rollout_k = self.rollout_top_k_train
      
      gt_goal = batch.extras['io_pairs_batch']['goal'][:, 0, :] # [B, N, 2]
      goal_dist = torch.norm(goal_point - gt_goal[:, :, None, :], dim=-1)
      _, goal_idxs = torch.min(goal_dist, dim=-1) # [B, N]
    
    else:
      rollout_k = min(self.rollout_top_k, K)
      _, goal_idxs_top_k = torch.topk(goal_prob, rollout_k, dim=-1)
      rand_idxs = torch.randint(0, rollout_k, (B, N,)).to(device=self.device)
      goal_idxs = torch.gather(goal_idxs_top_k, -1, rand_idxs[..., None]).squeeze(-1)

    policy_emds['select_idx'] = goal_idxs
    policy_emds['emd'] = torch.gather(policy_emds['emd'], -2, goal_idxs[..., None, None].repeat(1, 1, 1, D)).squeeze(-2)
    policy_emds['goal'] = torch.gather(goal_point, -2, goal_idxs[..., None, None].repeat(1, 1, 1, 2)).squeeze(-2)

    return policy_emds

  def _get_policy_batch_input(self, batch, policy_agent_ids, policy_emds, scene_embs, agent_positions, target_t):
    '''
    Organize the policy embedding, observation, and map for all agents in the batch.
    1. organize the policy embedding, observation, and map for all agents in the batch
    2. return the batch input for the policy decoder
    '''
    batch_policy_emd, batch_obs, batch_map, batch_pair_names, batch_pos = \
      [], [], [], [], []
    
    B = len(policy_agent_ids)
    
    bidxs = []
    tidxs = []
    nidxs = []
    batch_pair_names = []
    is_rollout = agent_positions is not None
  
    if not is_rollout:
      io_pair_batch = batch.extras['io_pairs_batch']
      if target_t is not None:
        target_tidx = io_pair_batch['T_indices'].index(target_t)

    num_agents = torch.tensor([len(agent_ids) for agent_ids in policy_agent_ids]).to(device=self.device) # [B]
    batch_ids = torch.arange(B, device=self.device)

    if is_rollout:
      N = max(num_agents)
      T = 1
      bidxs = batch_ids[:, None].repeat(1, N) # [B, N]
      nidxs = torch.arange(N, device=self.device)[None, :].repeat(B, 1) # [B, N]
      valid_mask = nidxs < num_agents[:, None] # [B, N]
      
      bidxs = bidxs[valid_mask].tolist()
      nidxs = nidxs[valid_mask]
      tidxs = torch.zeros_like(nidxs, device=self.device).tolist()  # Assuming tidx is 0 in this case
      nidxs = nidxs.tolist()

      batch_pair_names = [f'{bidx}-{policy_agent_ids[bidx][nidx]}-{target_t}' for bidx, nidx in zip(bidxs, nidxs)]

    else:
      N = io_pair_batch['mask'].shape[-1]
      assert N == max(num_agents)
      T = len(io_pair_batch['T_indices'])
      bidxs = batch_ids[:, None, None].repeat(1, T, N) # [B, T, N]
      nidxs = torch.arange(N, device=self.device)[None, None, :].repeat(B, T, 1) # [B, T, N]
      tidxs = torch.arange(T, device=self.device)[None, :, None].repeat(B, 1, N) # [B, T, N]

      n_valid_mask = nidxs < num_agents[:, None, None] # [B, T, N]
      data_valid_mask = io_pair_batch['mask'] # [B, T, N]
      valid_mask = n_valid_mask & data_valid_mask
      if target_t is not None:
        valid_mask = valid_mask & (tidxs == target_tidx)
      
      bidxs = bidxs[valid_mask].tolist()
      tidxs = tidxs[valid_mask].tolist()
      nidxs = nidxs[valid_mask].tolist()

      batch_pair_names = [f'{bidx}-{policy_agent_ids[bidx][nidx]}-{io_pair_batch["T_indices"][tidx]}' for bidx, nidx, tidx in zip(bidxs, nidxs, tidxs)]

    batch_policy_emd = {}
    for key in policy_emds.keys():
      if type(policy_emds[key]) == torch.Tensor:
        batch_policy_emd[key] = policy_emds[key][bidxs, nidxs]
      else:
        batch_policy_emd[key] = policy_emds[key]

    if is_rollout:
      batch_pos = {key: agent_positions[key][bidxs, nidxs] for key in ['position', 'heading']}
    else:
      batch_pos = {key: io_pair_batch[key][bidxs, tidxs, nidxs] for key in ['position', 'heading']}
    
    if 0 in scene_embs.keys():
      T_names = sorted(list(scene_embs.keys()))
      scene_embs_T = [scene_embs[t] for t in T_names]
    else:
      scene_embs_T = [scene_embs]
      tidxs = [0] * len(bidxs)
    
    batch_obs_T = self._scene_emd_to_batch(scene_embs_T, mode='obs')
    batch_map_T = self._scene_emd_to_batch(scene_embs_T, mode='map')

    batch_idx = torch.tensor(bidxs) * T + torch.tensor(tidxs)
    batch_policy_emd['batch_idx'] = batch_idx.to(self.device)

    return batch_policy_emd, batch_obs_T, batch_map_T, batch_pos, batch_pair_names

  def _visualize(self, batch, output, mode):
    for task in self.tasks:
      images = vis_scene_traj_pred(batch, output[task], self.config, target_T=int(0))

      if self.use_condition:
        batch_cond = batch.extras['condition']
        caption = ''
        for cond_type in batch_cond.keys():
          caption += f'{cond_type.upper()}: {batch_cond[cond_type]["caption_str"]}\n'
      else:
        caption = ''
          
      self._log_image(images, mode, '{}-pred'.format(task), caption=caption)

  def _update_scene_emb(self, last_scene_embs, batch_obs_new, old_obs_agent_ids):
    # clone the last_scene_embs
    scene_embs = {}
    for key in last_scene_embs:
      if type(last_scene_embs[key]) == torch.Tensor:
        scene_embs[key] = last_scene_embs[key].clone()
      else:
        scene_embs[key] = last_scene_embs[key]

    return self.scene_encoder.update_scene_emb(scene_embs, batch_obs_new, old_obs_agent_ids)
  
  def _get_rel_vel_acc(self, rel_xy, rel_vel=None):
    if rel_vel is None:
      rel_vel = torch.diff(rel_xy, dim=1) / self.config.DATASET.MOTION.DT # [N, T+1, 2] 

    rel_acc = torch.diff(rel_vel, dim=1) / self.config.DATASET.MOTION.DT # [N, T, 2]

    rel_vel_acc = torch.cat([rel_vel[:, 1:, :], rel_acc], dim=-1) # [N, T, 4]

    return rel_vel_acc

  def _process_rollout(self, agent_trajs, model_outputs, policy_agent_ids):
    '''
    Organize the rollout results for each agents in the batch for final output.
    '''
    task = self.tasks[0]
    
    result = {task: {}}

    for key in ['motion_pred', 'motion_prob', 'goal', 'pair_names', 'goal_prob', 'goal_point', 'select_idx', 'reconst_pred', 'prompt_loss']:
      if key not in model_outputs[0][task]:
        continue

      if key == 'pair_names':
        all_pair_names = [model_output[task][key] for model_output in model_outputs]
        result[task][key] = [item for sublist in all_pair_names for item in sublist]
      elif key == 'prompt_loss':
        result[task][key] = model_outputs[0][task][key]
      else:
        result[task][key] = torch.cat([model_output[task][key] for model_output in model_outputs], dim=0)
    
    result[task]['rollout_trajs'] = {}
    for bidx, agent_ids in enumerate(policy_agent_ids[task]):
      for nidx, agent_id in enumerate(agent_ids):
        aname = f'{bidx}-{agent_id}'
        result[task]['rollout_trajs'][aname] = {}

        result[task]['rollout_trajs'][aname]['traj'] = agent_trajs[task]['traj'][bidx, nidx, self.hist_step:]
        for key in ['init_pos', 'init_heading']:
          result[task]['rollout_trajs'][aname][key] = agent_trajs[task][key][bidx, nidx]
        
        if self.pred_vel:
          result[task]['rollout_trajs'][aname]['vel'] = agent_trajs[task]['vel'][bidx, nidx, self.hist_step:]

    return result

  def init_agent_trajs(self, policy_agent_ids, batch):
    '''
    Initialize the agent trajectories for all agents in the batch with initial observations.
    '''
    a_traj = {}

    for task in ['motion_pred']:
      a_traj[task] = {}
      task_policy_agent_ids = policy_agent_ids[task]
      init_obs = batch.extras['init_obs']

      B = len(task_policy_agent_ids)
      N = max([len(agent_ids) for agent_ids in task_policy_agent_ids])

      a_traj[task]['traj'] = torch.zeros(B, N, self.hist_step, 4, device=self.device)
      a_traj[task]['init_pos'] = torch.zeros(B, N, 2, device=self.device)
      a_traj[task]['init_heading'] = torch.zeros(B, N, 1, device=self.device)
      a_traj[task]['last_step'] = self.hist_step

      bidxs, nidxs, oidxs = [], [], []
      for bidx in range(B):
        for nidx, agent_id in enumerate(task_policy_agent_ids[bidx]):
          bidxs.append(bidx)
          nidxs.append(nidx)
          oidxs.append(init_obs['agent_ids'][bidx].index(agent_id))

      hist_traj = init_obs['input'][bidxs, oidxs, :, :4]
      a_traj[task]['traj'][bidxs, nidxs, :self.hist_step] = torch.nan_to_num(hist_traj, nan=0.0)
      a_traj[task]['init_pos'][bidxs, nidxs] = init_obs['position'][bidxs, oidxs]
      a_traj[task]['init_heading'][bidxs, nidxs] = init_obs['heading'][bidxs, oidxs, None]

      if self.pred_vel:
        hist_vel = init_obs['input'][bidxs, oidxs, :, 4:6]
        a_traj[task]['vel'] = torch.zeros(B, N, self.hist_step, 2, device=self.device)
        a_traj[task]['vel'][bidxs, nidxs, :self.hist_step] = torch.nan_to_num(hist_vel, nan=0.0)

    return a_traj

  def get_action(self, policy_emb, obs_data, map_data, pos_data, pair_names, latent_state=None):
    pair_names_all = []
    for pair_name in pair_names:
      pair_names_all += pair_name

    return self.policy(policy_emb, obs_data, map_data, pos_data, pair_names_all, latent_state)


  def _get_io_position(self, io_pair, agent_positions, pair_name):
    return self._get_io_position_from_dict(agent_positions, pair_name)