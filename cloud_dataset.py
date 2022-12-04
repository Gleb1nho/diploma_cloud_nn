import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
from PIL import Image


class CloudDataset(Dataset):
    def __init__(self, image_set):
        self.image_set = image_set

    def _merge_and_normalize(self, layers, index):
        return torch.from_numpy(
            cv2.normalize(
                cv2.merge(
                    [
                        cv2.cvtColor(cv2.imread(self.image_set[index][element]), cv2.COLOR_BGR2GRAY)
                        for element in layers
                    ]
                ),
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )
        ).permute(2, 0, 1)

    def __getitem__(self, index):
        image = self._merge_and_normalize(['red', 'green', 'blue'], index)
        mask = ToTensor()(Image.open(self.image_set[index]["mask"]).point(lambda p: p > 0.001 and 255))

        return image, mask

    def __len__(self):
        return len(list(self.image_set))


class ShowCloudDataset(Dataset):
    def __init__(self, image_set):
        self.image_set = image_set

    def __getitem__(self, index):
        raw = cv2.normalize(
                cv2.imread(self.image_set[index]),
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )

        nir = cv2.normalize(
            cv2.imread(self.image_set[index].replace('RGB', 'NIR'), 0),
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )

        # nir = cv2.imread(self.image_set[index].replace('RGB', 'NIR'), 0) / 255

        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        resized_rgb = cv2.resize(rgb, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        resized_nir = cv2.resize(nir, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        image = torch.from_numpy(np.dstack((resized_rgb, resized_nir))).permute(2, 0, 1)

        return image, resized_rgb, resized_nir

    def __len__(self):
        return len(list(self.image_set))

