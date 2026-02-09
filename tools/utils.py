import tensorflow.compat.v1 as tf
import os,cv2
import numpy as np
from tools.common_utils import *


def show_all_variables():
    model_vars = tf.trainable_variables()
    print('G:')
    for var in model_vars:
        if var.name.startswith('generator') and 'Adam' not in var.name:
            print(var.name)


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    """Defines gaussian kernel
    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        2-D Tensor of gaussian kernel
    """
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def gaussian_blur(img, kernel_size=7, sigma=5., ch = 3):
    """Convolves a gaussian kernel with input image
    Convolution is performed depthwise
    Args:
        img: 3-D Tensor of image, should by floats
        kernel: 2-D float Tensor for the gaussian kernel
    Returns:
        img: 3-D Tensor image convolved with gaussian kernel
    """
    blur = _gaussian_kernel(kernel_size, sigma, ch, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img

if __name__ == "__main__":
    path = '../dataset/val/1.jpg'
    image_foder = '../dataset/Hayao/style/11.jpg'
    Im = cv2.imread(image_foder).astype(np.float32)
    Im = np.expand_dims(Im, axis=0)
    a = gaussian_blur(tf.convert_to_tensor(Im))
    with tf.Session() as sess:
        S = sess.run(a)
    S = np.squeeze(S)
    # S=S.clip(0,1)
    print(type(S[0,0,0]),S.max(),S.min())
    cv2.imshow('a',S.astype(np.uint8))
    cv2.waitKey(0)
