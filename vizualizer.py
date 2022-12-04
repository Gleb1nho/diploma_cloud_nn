import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def visualize(**images):
    """Функция для визуализации данных, располагает изображения в ряд"""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show()


def show_model_results(test_set, best_model):
    """Функция для просмотра результатов работы"""
    best_model = torch.load(best_model)
    for i in range(10):
        n = np.random.choice(len(test_set))
        image, gt_mask = test_set[n]

        x_tensor = image.to('cuda').permute(0, 1, 2).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.detach().cpu().squeeze()

        visualize(
            image=image.permute(1, 2, 0),
            predicted=pr_mask,
            truth=gt_mask.squeeze()
        )


def show_large_image_results(test_set, best_model, tile_size=512):
    """Функция для просмотра результатов работы на снимках большего размера чем распознает обученная модель"""
    best_model = torch.load(best_model)

    for i in range(len(test_set)):
        image, rgb, nir = test_set[i]

        x_tensor = image.to('cuda').float()

        h = x_tensor.shape[1]
        w = x_tensor.shape[2]

        result = np.zeros((1, h, w), dtype=np.float32)

        full_height_tiles = [x * tile_size for x in range(h // tile_size)]
        full_width_tiles = [x * tile_size for x in range(w // tile_size)]

        if h % tile_size != 0:
            full_height_tiles.append(h - tile_size)

        if w % tile_size != 0:
            full_width_tiles.append(w - tile_size)

        for k in full_height_tiles:
            for j in full_width_tiles:
                tile = x_tensor[:, k:k+tile_size, j:j+tile_size]

                predicted = best_model.predict(tile.unsqueeze(0)).detach().cpu().squeeze()

                result[:, k:k + tile_size, j:j + tile_size] = predicted

        result_image = Image.fromarray(np.uint8(np.reshape(result, (h, w)) * 255), 'L')

        visualize(
            # image=image.permute(1, 2, 0),
            rgb=rgb,
            nir=nir,
            predicted=result_image
        )
