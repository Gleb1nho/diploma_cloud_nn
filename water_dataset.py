import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch

from diploma_cloud_nn.cloud_dataset import ShowCloudDataset
from vizualizer import visualize, show_large_image_results

from diploma_cloud_nn.water_trainer import WaterDetectorLearner

images_dir = '../SWED/train/images'
masks_dir = '../SWED/train/labels'

with open(f'./correct_water_filenames', 'r') as filenames:
    names = list(set(filenames.readlines()))
    train_sett = [
        {'image': f'{images_dir}/{name.strip()}', 'mask': f'{masks_dir}/{name.strip().replace("image", "chip")}'} for
        name in names[0:20000]]
    valid_set = [
        {'image': f'{images_dir}/{name.strip()}', 'mask': f'{masks_dir}/{name.strip().replace("image", "chip")}'} for
        name in names[20000:25000]]


class WaterDataset(Dataset):
    def __init__(self, imageset):
        self.image_set = imageset

    def __getitem__(self, index):
        image = np.load(self.image_set[index]["image"])

        mask = torch.from_numpy(np.load(self.image_set[index]["mask"]))

        r = (image[:, :, 3] / (np.max(image[:, :, 3]))).astype(np.float32)
        g = (image[:, :, 2] / (np.max(image[:, :, 2]))).astype(np.float32)
        b = (image[:, :, 1] / (np.max(image[:, :, 1]))).astype(np.float32)
        nir = (image[:, :, 7] / (np.max(image[:, :, 7]))).astype(np.float32)

        image = torch.from_numpy(np.dstack((r, g, b, nir))).permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(list(self.image_set))


train_set = WaterDataset(train_sett)
valid_set = WaterDataset(valid_set)


def test_show(index):
    best_model = torch.load('../water_exp2_dice_loss_best_model.pth')
    image, gt_mask = train_set[index]

    img = np.load(train_sett[index]['image'])

    r = (img[:, :, 3] / (np.max(img[:, :, 3]))).astype(np.float32)
    g = (img[:, :, 2] / (np.max(img[:, :, 2]))).astype(np.float32)
    b = (img[:, :, 1] / (np.max(img[:, :, 1]))).astype(np.float32)
    nir = (img[:, :, 7] / (np.max(img[:, :, 7]))).astype(np.float32)

    x_tensor = image.to('cuda').permute(0, 1, 2).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = pr_mask.detach().cpu().squeeze()

    visualize(
        rgb=np.dstack((r, g, b)),
        nir=nir,
        predicted=pr_mask,
        truth=gt_mask.squeeze()
    )

# for i in range(10):
#     test_show(i)

# plt.imshow()
# plt.show()
#
# plt.imshow(pr_mask)
# plt.show()
#
# plt.imshow(gt_mask.squeeze())
# plt.show()


# train_set.__getitem__(0)
#
# trainer = WaterDetectorLearner(
#     train_set,
#     valid_set
# )
#
# if __name__ == '__main__':
#     trainer.start_training()
