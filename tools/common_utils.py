import cv2, os
import numpy as np

def img_resize(img, limit=1280):
    h, w = img.shape[:2]
    max_edge = max(h,w)
    if max_edge > limit:
        scale_factor = limit / max_edge
        height = int(round(h * scale_factor))
        width = int(round(w * scale_factor))
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    return img


def load_test_data(image_path, x8=True):
    img = cv2.imread(image_path)#.astype(np.float32)
    img = img_resize(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,x8)
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8: # resize image to multiple of 8s
        def to_8s(x):
            return 256 if x < 256 else x - x%8  # if using tiny model: x - x%16
        img = cv2.resize(img, (to_8s(w), to_8s(h)))
    return img/127.5 - 1.0

def save_images(images, image_path):
    fake = inverse_transform(images.squeeze())
    return imsave(fake, image_path)

def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate, 
    # and the range of pixel values must be limited to the boundary, 
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')
