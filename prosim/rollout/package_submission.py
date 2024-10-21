import os
import tqdm
import glob
from pathlib import Path
import tarfile
import argparse
from multiprocessing import Pool

from waymo_open_dataset.protos.sim_agents_submission_pb2 import SimAgentsChallengeSubmission
from waymo_open_dataset.protos.sim_agents_submission_pb2 import ScenarioRollouts

import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from prosim.rollout.baseline import rollout_baseline, get_waymo_scene_object

import psutil

def print_system_mem_usage():
  # Get the memory details
  memory = psutil.virtual_memory()

  # Total memory
  total_memory = memory.total

  # Available memory
  available_memory = memory.available

  # Used memory
  used_memory = memory.used

  # Memory usage percentage
  memory_percent = memory.percent

  print(f"Total Memory: {total_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Available Memory: {available_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Used Memory: {used_memory / (1024 * 1024 * 1024):.2f} GB", flush=True)
  print(f"Memory Usage: {memory_percent}%", flush=True)


argparser = argparse.ArgumentParser()
argparser.add_argument('--root', type=str, required=True)
argparser.add_argument('--mode', type=str, default='val')
argparser.add_argument('--baseline', type=bool, default=False)
argparser.add_argument('--num_workers', type=int, default=150)
argparser.add_argument('--method_name', type=str, default='debug')
args = argparser.parse_args()

save_root = Path(args.root)

rollout_folder = save_root / 'rollout'
rollout_files = glob.glob(str(rollout_folder / '*.pb'))

if args.mode == 'val':
  mode_name = 'validation'
elif args.mode == 'test':
  mode_name = 'testing'
scene_template = Path('/lustre/fsw/portfolios/nvr/users/shuhant/waymo_v_1_2_0/scenario') / f'{mode_name}_splitted'
file_template = f'{mode_name}_splitted_' + '{}.tfrecords'
scene_template = scene_template / file_template

if args.mode == 'val':
   assert len(rollout_files) == 44097

def load_rollout_from_file(file_name):
  with open(file_name, 'rb') as f:
    string = f.read()
  return ScenarioRollouts.FromString(string)

def run_baseline_rollout(scene_id):
  file_path = str(scene_template).format(scene_id)
  file_path = Path(file_path)
  file_path = os.path.expanduser(file_path)
  waymo_scene = get_waymo_scene_object(file_path)
  rollout = rollout_baseline(waymo_scene)

  return rollout

def get_scene_id_from_file(file):
  return int(file.split('/')[-1].split('.')[0].split('_')[-1])

def pakage_submission_file(worker_id):
    print(f'worker {worker_id} started!\n')
    worker_files = []
    for file in rollout_files:
        scene_id = get_scene_id_from_file(file)
        if scene_id % num_workers == worker_id:
            worker_files.append(file)
    
    scenario_rollouts = []

    if args.baseline:
      print('running baseline!')
    else:
      print('loading rollouts!')
    
    for file in tqdm.tqdm(worker_files, desc=f'worker {worker_id}'):
        if args.baseline:
          scene_id = get_scene_id_from_file(file)
          rollout = run_baseline_rollout(scene_id)
        else:
          rollout = load_rollout_from_file(file)
        print_system_mem_usage()
        scenario_rollouts.append(rollout)
    
    unique_method_name = 'extrapolate_baseline' if args.baseline else args.method_name

    shard_submission = SimAgentsChallengeSubmission(
          scenario_rollouts=scenario_rollouts,
          submission_type=SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
          account_name='shuhan@utexas.edu',
          unique_method_name=unique_method_name,
          authors=['shuhant'],
          affiliation='utexas',
          description='null',
          method_link='https://waymo.com/open/'
      )
    
    output_file_name = submission_folder / f'submission.binproto-{worker_id:05d}-of-{num_workers:05d}'
    
    with open(output_file_name, 'wb') as f:
        f.write(shard_submission.SerializeToString())
    
    return output_file_name

num_workers = args.num_workers

if args.baseline:
  submission_folder = save_root / 'baseline_submission'
else:
  submission_folder = save_root / 'submission'
submission_folder.mkdir(parents=True, exist_ok=True)

submission_tar = submission_folder / 'submission.tar.gz'

# Create a pool of workers
with Pool(num_workers) as pool:
    # Distribute the work among the workers
    file_names = pool.map(pakage_submission_file, range(num_workers))

# Once we have created all the shards, we can package them directly into a
# tar.gz archive, ready for submission.
with tarfile.open(submission_tar, 'w:gz') as tar:
    for file_name in file_names:
      tar.add(file_name, arcname=file_name.name)