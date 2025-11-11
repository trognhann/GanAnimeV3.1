import os
import re
import yaml

import yaml

with open('./config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATASET = config['DATASET']
DATA_TEMP = config['DATA_TEMP']
DATA_ANIME = config['DATA_ANIME']
DATASET_SMOOTH = config['DATASET_SMOOTH']
DATASET_SMOOTH_NOISE = config['DATASET_SMOOTH_NOISE']
