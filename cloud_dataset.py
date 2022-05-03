import cv2
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

