import mmcv
import numpy as np
import imageio
import cv2


palette = np.array([[255, 255, 255], [0, 0, 0]])


def rgb_to_gray(arr_3d, palette=palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    i = 0
    for c in palette:
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
        i += 1

    return arr_2d


def extract_blues(mask):
    # Это не красный а синий, на самом деле :)
    mask_water_color = np.array([255, 0, 0], dtype='uint8')

    blue_mask = cv2.inRange(mask, mask_water_color, mask_water_color)
    extracted_blues = cv2.bitwise_and(mask, mask, mask=blue_mask)

    return extracted_blues


def extract_water_from_masks(masks_file='train_masks.txt', path_to_images='../../Projects/DeepGlobe/train'):
    masks = []

    with open(masks_file, 'r') as images_file:
        for item in map(lambda item: f'{path_to_images}/{item.strip()}', images_file.readlines()):
            masks.append(item)

    for item in masks:
        read_mask = mmcv.imread(item)
        extracted_blues = extract_blues(read_mask)
        converted_mask = rgb_to_gray(extracted_blues)
        converted_mask = ((converted_mask - 1) ** 2) * 255

        imageio.imwrite(item, converted_mask)


if __name__ == '__main__':
    extract_water_from_masks()
