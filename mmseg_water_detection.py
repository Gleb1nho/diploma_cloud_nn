import os.path as osp
import numpy as np
from PIL import Image
import mmcv


data_root = 'train'
img_dir = ''
ann_dir = ''
classes = ('water', 'notwater')
palette = [[255, 255, 255], [0, 0, 0]]

masks_file = 'train_masks.txt'
images_file = 'train_images.txt'
path_to_images = '../../Projects/DeepGlobe/train'


def read_images_from_file(file, images_path='../../Projects/DeepGlobe/train'):
    result = []

    with open(file, 'r') as images_file:
        for item in map(lambda item: f'{images_path}/{item.strip()}', images_file.readlines()):
            result.append(item)

    return result


images = read_images_from_file(images_file)
masks = read_images_from_file(masks_file)
