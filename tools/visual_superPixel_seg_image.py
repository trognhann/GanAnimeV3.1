import cv2
import os
import numpy as np
from skimage import segmentation, color
from tqdm import tqdm

from .utils import check_folder
import config as cfg


def get_simple_superpixel_improve(image_path, seg_num=200):
    """
    Segment an image into superpixels using SLIC and produce a smoothed image
    by filling each superpixel with its average color.

    Parameters
    ----------
    image_path : str
        Path to the input image file to read (expected readable by cv2.imread).
    seg_num : int, optional
        Target number of superpixels to generate (default is 200).

    Returns
    -------
    numpy.ndarray
        RGB image (H x W x 3) where each superpixel region has been replaced by
        its mean color. The result appears smoother with reduced fine-grained
        noise and is often easier to process for downstream tasks such as
        recognition or classification.

    Notes
    -----
    - Internally, SLIC is used to compute superpixel labels and color.label2rgb
      is used to compute the average color per label.
    - If the image cannot be read (cv2.imread returns None), behavior is
      implementation-defined; callers should ensure the path is valid.

    Examples
    --------
    >>> img = get_simple_superpixel_improve("/path/to/image.jpg", seg_num=250)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1, start_label=0,
                                  compactness=10, convert2lab=True)
    image = color.label2rgb(seg_label, image, bg_label=-1, kind='avg')
    return image


def get_superPixel(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rgb = cv2.resize(rgb, (256,256))
    # img_seg = segmentation.felzenszwalb(rgb, scale=5, sigma=0.8, min_size=100) # photo_superpixel
    # img_seg = segmentation.felzenszwalb(rgb, scale=200, sigma=0.5, min_size=200)
    img_seg = segmentation.felzenszwalb(rgb, scale=5, sigma=0.8, min_size=50)
    out = color.label2rgb(img_seg, rgb, bg_label=-1, kind='avg')
    return out


def main1():
    check_folder(cfg.DATA_TEMP)

    for i, x in tqdm(enumerate(os.listdir(cfg.DATA_ANIME))):
        print(i, x)
        path = os.path.join(cfg.DATA_ANIME, x)
        img = get_superPixel(path)
        # img = get_simple_superpixel_improve(path, 1000)
        cv2.imwrite(os.path.join(cfg.DATA_TEMP, x), cv2.cvtColor(
            img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.imshow('super_seg',cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)


if __name__ == '__main__':
    main1()
