import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import random

import numpy as np
import torch
import wandb

from prosim.config.default import Config, get_config
from prosim.trainer import BaseTrainer as Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "data_debug"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--cluster",
        default='local', # 'local', 'ngc', 'slurm',
        type=str,
        help="which cluster to run on",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    trainer = Trainer(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == 'data_debug':
        trainer.data_debug()
    
    if config.LOGGER == 'wandb':
        wandb.finish()

    return trainer.save_dir

def run_exp(exp_config: str, run_type: str, opts, cluster) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts, cluster)
    execute_exp(config, run_type)

if __name__ == "__main__":
    main()