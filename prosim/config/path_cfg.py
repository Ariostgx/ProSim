import os
from typing import List, Optional, Union
import yacs.config

class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)
CN = Config

# obtain the directory to this file
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(os.path.dirname(code_dir), 'demo_dataset')
label_dir = os.path.dirname(os.path.dirname(code_dir))

ROOT_DIR = code_dir
TRAJDATA_CACHE_DIR = os.path.join(data_dir, 'trajdata_cache')
WAYMO_DATA_DIR = os.path.join(data_dir, 'waymo_source')
DEMO_DATA_DIR = TRAJDATA_CACHE_DIR
# LLM_MODEL_DIR = '/storage/Models/shuhan/Meta-Llama-3-8B-Instruct'
LLM_MODEL_DIR = '' # Replace with your local path to Meta-Llama-3-8B-Instruct huggingface model directory

PROSIM_INSTRUCT_520K_DATA_DIR = os.path.join(label_dir, 'prosim_instruct_520k')

MOTION_TAG_PATH = os.path.join(PROSIM_INSTRUCT_520K_DATA_DIR, 'tag_prompts')
TEXT_PROMPT_PATH = os.path.join(PROSIM_INSTRUCT_520K_DATA_DIR, 'text_prompts')
RESULT_PATH = os.path.join(ROOT_DIR, 'result')

print(ROOT_DIR)
print(data_dir)

PATHS = {}

PATHS['local'] = CN()
PATHS['local'].DATASET = CN()
PATHS['local'].DATASET.DATA_LIST = CN()

PATHS['local'].ROOT_DIR = ROOT_DIR
PATHS['local'].DATASET.DATA_LIST.ROOT = os.path.join(ROOT_DIR, 'data_list')

PATHS['local'].DATASET.DATA_PATHS = CN()

PATHS['local'].DATASET.DATA_PATHS.WAYMO_TRAIN = WAYMO_DATA_DIR
PATHS['local'].DATASET.DATA_PATHS.WAYMO_VAL = WAYMO_DATA_DIR
PATHS['local'].DATASET.DATA_PATHS.WAYMO_TEST = WAYMO_DATA_DIR
PATHS['local'].DATASET.DATA_PATHS.WAYMO_ROAD_EDGE_CACHE = None
PATHS['local'].DATASET.DATA_PATHS.VECTOR_LANE_CACHE = os.path.join(DEMO_DATA_DIR, 'input_vec_map_cache')

PATHS['local'].DATASET.DATA_PATHS.TRAIN_DRIVESIM_MAIN = DEMO_DATA_DIR

PATHS['local'].MODEL = CN()

PATHS['local'].MODEL.CONDITION_TRANSFORMER = CN()
PATHS['local'].MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER = CN()
PATHS['local'].MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT = CN()
PATHS['local'].MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM = CN()
PATHS['local'].MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH = CN()
PATHS['local'].MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH.LLAMA3_8B_INSTRUCT = LLM_MODEL_DIR

PATHS['local'].MODEL.POLICY = CN()
PATHS['local'].MODEL.POLICY.ACT_DECODER = CN()
PATHS['local'].MODEL.POLICY.ACT_DECODER.TRAJ = CN()
PATHS['local'].MODEL.POLICY.ACT_DECODER.TRAJ.CLUSTER_PATH = ''

PATHS['local'].MODEL.DECODER = CN()
PATHS['local'].MODEL.DECODER.GOAL_PRED = CN()
PATHS['local'].MODEL.DECODER.GOAL_PRED.K_CLUSTER_PATH = None

PATHS['local'].DATASET.CACHE_PATH = TRAJDATA_CACHE_DIR
PATHS['local'].DATASET.DATA_PATHS.PROSIM_INSTRUCT_520K = PROSIM_INSTRUCT_520K_DATA_DIR
PATHS['local'].DATASET.DATA_PATHS.MOTION_TAGS = CN()
PATHS['local'].DATASET.DATA_PATHS.MOTION_TAGS.TRAIN = os.path.join(MOTION_TAG_PATH, 'waymo_train_v_action/tags')
PATHS['local'].DATASET.DATA_PATHS.MOTION_TAGS.VAL = os.path.join(MOTION_TAG_PATH, 'waymo_val_v_action/tags')
PATHS['local'].DATASET.DATA_PATHS.MOTION_TAGS.TEST = os.path.join(MOTION_TAG_PATH, 'waymo_val_v_action/tags')

PATHS['local'].SAVE_DIR = RESULT_PATH
PATHS['local'].LOGGER = 'wandb'

PATHS['local'].PROMPT = CN()
PATHS['local'].PROMPT.CONDITION = CN()
PATHS['local'].PROMPT.CONDITION.LLM_TEXT_FOLDER = CN()
PATHS['local'].PROMPT.CONDITION.LLM_TEXT_FOLDER.TRAIN = os.path.join(TEXT_PROMPT_PATH, 'train/70b_0.8_0.9_action_prompt_v2')
PATHS['local'].PROMPT.CONDITION.LLM_TEXT_FOLDER.VAL = os.path.join(TEXT_PROMPT_PATH, 'val/70b_0.8_0.9_action_prompt_v2')
PATHS['local'].PROMPT.CONDITION.LLM_TEXT_FOLDER.TEST = os.path.join(TEXT_PROMPT_PATH, 'val/70b_0.8_0.9_action_prompt_v2')
