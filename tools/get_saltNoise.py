import os

import numpy as np
import cv2
from tqdm import tqdm

from .utils import check_folder
from config import *

opj = os.path.join


def sp_noise(img, prob=0.4):
    '''
    Add salt and pepper noise
     prob: noise ratio
    '''
    output = img.copy()
    h, w = output.shape[:2]
    sp = h*w
    NP = int(sp*(prob))
    for i in range(NP):
        randy = np.random.randint(1, h-1)
        randx = np.random.randint(1, w-1)
        output[randy, randx] = 255
    return output


def main2():
    check_folder(DATASET_SMOOTH_NOISE)
    imgs = os.listdir(DATASET_SMOOTH)
    for x in tqdm(imgs):
        img = cv2.imread(opj(DATASET_SMOOTH, x))
        S = sp_noise(img, 0.1)
        cv2.imwrite(opj(DATASET_SMOOTH_NOISE, x), S)

if __name__ == "__main__":
    image_foder = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3/dataset/Shinkai/smooth'
    out_foder = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3/dataset/Shinkai/smooth_noise'
    check_folder(out_foder)
    imgs = os.listdir(image_foder)
    for x in tqdm(imgs):
        img = cv2.imread(opj(image_foder, x))
        S = sp_noise(img, 0.1)
        cv2.imwrite(opj(out_foder, x), S)
