import os
import re
import yaml

import yaml

with open('./config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATASET = config['DATASET']
print("Loaded dataset config:", DATASET)
