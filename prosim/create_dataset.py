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

from torch.utils.data import DataLoader

from prosim.config.default import Config, get_config
from prosim.core.registry import registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    task_config = config

    dataset_configs = {'train': config.TRAIN,
                           'val': config.VAL, 'test': config.TEST}
    dataset_type = task_config.DATASET.TYPE

    data_loaders = {}
    for mode, config in dataset_configs.items():
      dataset = registry.get_dataset(dataset_type)(task_config, config.SPLIT)
      batch_size = config.BATCH_SIZE
      data_loaders[mode] = DataLoader(dataset, batch_size=batch_size, shuffle=config.SHUFFLE, pin_memory=True, drop_last=config.DROP_LAST, num_workers=config.NUM_WORKERS, collate_fn=dataset.get_collate_fn())

def run_exp(exp_config: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)
    execute_exp(config)

if __name__ == "__main__":
    main()