import matplotlib.pyplot as plt
import numpy as np
import torch


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


def show_model_results(test_set):
    """Функция для просмотра результатов работы"""
    best_model = torch.load('./best_model.pth')
    for i in range(1):
        n = np.random.choice(len(test_set))
        image, gt_mask = test_set[n]

        x_tensor = image.to('cuda').permute(0, 1, 2).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.detach().cpu().squeeze()

        # Нормализация, возможно понадобится
        # pr_mask = torch.nn.functional.normalize(pr_mask)\
        #     .apply_(lambda p: p > 0.001 and 1)

        visualize(
            image=image.permute(1, 2, 0),
            predicted=pr_mask,
            truth=gt_mask.squeeze()
        )

