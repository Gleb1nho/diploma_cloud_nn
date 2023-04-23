import pickle
import os
from PIL import Image


def cut_image_in_tiles(width: int, height: int, tile_width: int, tile_height: int, offset: int) -> list[dict[str, int]]:
    result = []

    x_tiles = []
    y_tiles = []

    current_width = 0

    while current_width + tile_width < width:
        x_tiles.append(
            {'left': current_width, 'right': current_width + tile_width})

        current_width += offset

    last_width_tile = {'left': width - tile_width, 'right': width}

    if last_width_tile not in x_tiles:
        x_tiles.append(last_width_tile)

    current_height = 0

    while current_height + tile_height < height:
        y_tiles.append(
            {'top': current_height, 'bottom': current_height + tile_height})

        current_height += offset

    last_height_tile = {'top': height - tile_height, 'bottom': height}

    if last_height_tile not in y_tiles:
        y_tiles.append(last_height_tile)

    for x_tile in x_tiles:
        for y_tile in y_tiles:
            result.append({'left': x_tile['left'], 'right': x_tile['right'],
                          'top': y_tile['top'], 'bottom': y_tile['bottom']})

    return result


def get_images_and_masks() -> dict[str, dict[str, str]]:
    if not os.path.exists('images_and_masks.pickle'):
        images_and_masks = {}

        with open('train_images.txt', 'r') as images:
            for image in images.readlines():
                split_image, _ = image.split('_')
                images_and_masks[split_image] = {
                    'image': f'{split_image}_sat.jpg'}

        with open('train_masks.txt', 'r') as masks:
            for mask in masks.readlines():
                split_mask, _ = mask.split('_')
                images_and_masks[split_mask]['mask'] = f'{split_mask}_mask.png'

        with open('images_and_masks.pickle', 'wb') as serialized:
            pickle.dump(images_and_masks, serialized)

    with open('images_and_masks.pickle', 'rb') as serialized:
        deserialized = pickle.load(serialized)

    return deserialized


def get_cut_images_and_masks() -> dict[str, dict[str, str]]:
    if not os.path.exists('cut_images_and_masks.pickle'):
        images_and_masks = {}

        with open('cut_images.txt', 'r') as images:
            for image in images.readlines():
                split_image = image.replace('_sat.jpg', '').strip()
                images_and_masks[split_image] = {
                    'image': f'{split_image}_sat.jpg'}

        with open('cut_masks.txt', 'r') as masks:
            for mask in masks.readlines():
                split_mask = mask.replace('_mask.png', '').strip()
                images_and_masks[split_mask]['mask'] = f'{split_mask}_mask.png'

        with open('cut_images_and_masks.pickle', 'wb') as serialized:
            pickle.dump(images_and_masks, serialized)

    with open('cut_images_and_masks.pickle', 'rb') as serialized:
        deserialized = pickle.load(serialized)

    return deserialized


def crop_images(images_path: str, target_dir: str, tiles: list[dict[str, int]], images_and_masks: dict[str, dict[str, str]]) -> None:
    for image_and_mask in images_and_masks:
        image, mask = images_and_masks[image_and_mask]['image'], images_and_masks[image_and_mask]['mask']

        read_image = Image.open(f'{images_path}/{image}')
        read_mask = Image.open(f'{images_path}/{mask}')

        for index in range(0, len(tiles)):
            tile = tiles[index]

            left = tile['left']
            right = tile['right']
            top = tile['top']
            bottom = tile['bottom']

            cropped_image = read_image.crop((left, top, right, bottom))
            cropped_mask = read_mask.crop((left, top, right, bottom))

            cropped_image.save(f'{target_dir}/{index}_{image}')
            cropped_mask.save(f'{target_dir}/{index}_{mask}')
