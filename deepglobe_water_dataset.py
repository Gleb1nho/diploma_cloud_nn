from deepglobe_image_cutter import get_cut_images_and_masks
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

from water_trainer import WaterDetectorLearner


images_path = '../../Projects/cut_deepglobe'


images_and_masks = get_cut_images_and_masks()

list_keys = []

for key in images_and_masks:
    list_keys.append(key)

list_keys = list(set(list_keys))

all_images_and_masks = []

for key in list_keys:
    mask = images_and_masks[key]['mask']
    image = images_and_masks[key]['image']

    all_images_and_masks.append({
        'image': f'{images_path}/{image}',
        'mask': f'{images_path}/{mask}'
    })

train_set = all_images_and_masks[0:190000]
valid_set = all_images_and_masks[190000:]


class DeepGlobeWaterDataset(Dataset):
    def __init__(self, image_set):
        self.image_set = image_set

    def __getitem__(self, index):
        image = ToTensor()(Image.open(self.image_set[index]["image"]))
        mask = ToTensor()(Image.open(self.image_set[index]["mask"]))

        return image, mask

    def __len__(self):
        return len(list(self.image_set))


train_set = DeepGlobeWaterDataset(train_set)
valid_set = DeepGlobeWaterDataset(valid_set)

trainer = WaterDetectorLearner(
    train_set,
    valid_set
)

trainer.start_training()
