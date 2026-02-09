import json
import os
import re


def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""


def clean_code(code, removed_imports):
    lines = code.split('\n')
    cleaned = []
    for line in lines:
        skip = False
        for imp in removed_imports:
            if line.strip().startswith(imp):
                skip = True
                break
        if not skip:
            cleaned.append(line)
    return '\n'.join(cleaned)


cells = []

# 1. Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# AnimeGANv3 Consolidated Notebook\n",
        "This notebook consolidates the AnimeGANv3 project into a single runnable file.\n",
        "Includes Training, Inference, and Model definitions."
    ]
})

# 2. Setup
setup_code = """
import sys
import time
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import logging
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import skimage
from skimage import segmentation, color
from skimage.io import imread, imsave
from skimage import img_as_float
import argparse
import random

# Configuration
try:
    tf.disable_v2_behavior()
except ImportError:
    print("TensorFlow 1.x logic might need tf.compat.v1")

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.test.is_gpu_available()}")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code.splitlines(True)
})

# List of files to merge and their titles
# Order matters!
removed_imports = [
    "from tools.utils import", "from tools.common_utils import", "from tools.tf_color_ops import",
    "from tools.L0_smoothing import", "from tools.GuidedFilter import", "from tools.ops import",
    "from tools import", "from net import", "from AnimeGANv3_shinkai import",
    "import tensorflow.compat.v1 as tf", "import cv2", "import os", "import numpy as np"
]

files_to_merge = [
    ("tools/utils.py", "Utils"),
    ("tools/common_utils.py", "Common Utils"),
    ("tools/tf_color_ops.py", "TF Color Ops"),
    ("tools/L0_smoothing.py", "L0 Smoothing"),
    ("tools/GuidedFilter.py", "Guided Filter"),
    ("tools/vgg19.py", "VGG19"),
    ("tools/ops.py", "Ops"),
    ("tools/data_loader.py", "Data Loader"),
    ("net/generator.py", "Generator Network"),
    ("net/discriminator.py", "Discriminator Network"),
    ("AnimeGANv3_shinkai.py", "AnimeGANv3 Model Class")
]

for rel_path, title in files_to_merge:
    full_path = os.path.join(os.getcwd(), rel_path)
    content = read_file(full_path)
    cleaned_content = clean_code(content, removed_imports)

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {title}"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cleaned_content.splitlines(True)
    })

# Special handling for Train and Test to avoid auto-execution
train_content = read_file("train.py")
train_content = clean_code(train_content, removed_imports)
train_content = train_content.replace(
    "if __name__ == '__main__':", "# if __name__ == '__main__':")
train_content = train_content.replace(
    "train()", "# train()")  # Prevent auto run

# Add a Config class to replace argparse in Train
config_cell = """
class TrainConfig:
    def __init__(self):
        self.style_dataset = 'Hayao'
        self.init_G_epoch = 1
        self.epoch = 10
        self.batch_size = 4
        self.save_freq = 1
        self.load_or_resume = 'load'
        self.init_G_lr = 2e-4
        self.g_lr = 1e-4
        self.d_lr = 1e-4
        self.img_size = [256, 256]
        self.img_ch = 3
        self.sn = True
        self.checkpoint_dir = 'checkpoint'
        self.log_dir = 'logs'
        self.sample_dir = 'samples'

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Overriding parse_args for notebook usage
def parse_args():
    return TrainConfig()
"""

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Training Logic"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": config_cell.splitlines(True)
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": train_content.splitlines(True)
})

# Test/Inference
test_content = read_file("test.py")
test_content = clean_code(test_content, removed_imports)
test_content = test_content.replace(
    "if __name__ == '__main__':", "# if __name__ == '__main__':")
test_content = test_content.replace("test(", "# test(")

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Inference/Test Logic"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": test_content.splitlines(True)
})

# Sample Execution Cell
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Run Example"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Example usage:\n",
        "# train() \n",
        "# Or for inference:\n",
        "# test('checkpoint/generator_v3_Hayao_weight', 'style_results/', 'inputs/imgs')\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("train.ipynb generated.")
