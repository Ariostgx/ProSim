import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, required=True)
argparser.add_argument("--ckpt", type=str, required=True)
argparser.add_argument('--rollout_name', type=str, required=True)
argparser.add_argument('--save_metric', type=bool, default=True)
argparser.add_argument('--save_rollout', type=bool, default=True)
argparser.add_argument('--cluster', type=str, default='local')
argparser.add_argument("--M", type=int, default=32)
argparser.add_argument("--action_noise_std", type=float, default=0.0)
argparser.add_argument("--traj_noise_std", type=float, default=0.0)
argparser.add_argument("--top_k", type=int, default=3)
argparser.add_argument("--smooth_dist", type=float, default=5.0)
argparser.add_argument("--sampler_cfg", type=str, default=None)

args = argparser.parse_args()

from prosim.core.registry import registry
from prosim.config.default import Config, get_config
from prosim.rollout.distributed_utils import rollout_scene_distributed

print(args.cluster)

print('save_metric: ', args.save_metric)
print('save_rollout: ', args.save_rollout)

config = get_config(args.config, cluster=args.cluster)
rollout_scene_distributed(config, args.M, args.ckpt, args.rollout_name, args.save_metric, args.save_rollout, args.top_k, args.traj_noise_std, args.action_noise_std, args.sampler_cfg, args.smooth_dist)