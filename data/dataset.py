import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A
from torchvision.transforms.v2.functional import crop


@dataclass
class ImagewoofDatasetInfo:
    image_size = 256
    num_classes = 10
    id2class = ['Shih-Tzu', 'Rhodesian_ridgeback', 'Beagle', 'English_foxhound', 'Australian_terrier',
                'Border_terrier', 'Golden_retriever', 'Old_English_sheepdog', 'Samoyed', 'Dingo']
    class2id = {'Shih-Tzu': 0, 'Rhodesian_ridgeback': 1, 'Beagle': 2, 'English_foxhound': 3, 'Australian_terrier': 4,
                'Border_terrier': 5, 'Golden_retriever': 6, 'Old_English_sheepdog': 7, 'Samoyed': 8, 'Dingo': 9}
    dir_name2class = {'n02086240': 'Shih-Tzu', 'n02087394': 'Rhodesian_ridgeback', 'n02088364': 'Beagle',
                      'n02089973': 'English_foxhound', 'n02093754': 'Australian_terrier', 'n02096294': 'Border_terrier',
                      'n02099601': 'Golden_retriever', 'n02105641': 'Old_English_sheepdog', 'n02111889': 'Samoyed',
                      'n02115641': 'Dingo'}


class ImagewoofDataset(Dataset):
    @staticmethod
    def get_info():
        return ImagewoofDatasetInfo()

    def __init__(self, data_dir, ratio, train):
        self.info = self.get_info()

        self.train = train
        if self.train:
            data_dir /= 'train'
        else:
            data_dir /= 'val'

        self.image_paths = []
        self.class_ids = []
        for dir_name, class_name in self.info.dir_name2class.items():
            class_dir = data_dir / dir_name
            image_paths = sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                 key=lambda image_path: image_path.name)
            class_id = self.info.class2id[class_name]
            for image_path in image_paths[:int(len(image_paths) * ratio)]:
                self.image_paths.append(image_path)
                self.class_ids.append(class_id)

        self.train_transforms = A.Compose([
            A.RandomResizedCrop(height=self.info.image_size, width=self.info.image_size, scale=(0.6, 1)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            A.Resize(height=self.info.image_size, width=self.info.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pil_image = ImageOps.exif_transpose(Image.open(self.image_paths[index]).convert('RGB'))
        transforms = self.train_transforms if self.train else self.val_transforms
        transformed_image = transforms(image=np.array(pil_image))['image']
        return transformed_image, self.class_ids[index]


class ImagewoofVCPDataset(Dataset):
    def __init__(self, data_dir):
        self.info = ImagewoofDataset.get_info()

        self.image_paths = []
        for dataset_type in ('train', 'val'):
            for dir_name in self.info.dir_name2class.keys():
                class_dir = data_dir / dataset_type / dir_name
                image_paths = sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                     key=lambda image_path: image_path.name)
                self.image_paths.extend(image_paths)

        self.transforms = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.patch_size = 512 // 3

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pil_image = ImageOps.exif_transpose(Image.open(self.image_paths[index]).convert('RGB'))
        image = self.transforms(image=np.array(pil_image))['image']

        center_patch = crop(image, top=self.patch_size, left=self.patch_size, height=self.patch_size,
                            width=self.patch_size)

        patch_id = random.randint(0, 8)

        patch_crop = crop(image, top=self.patch_size * (patch_id // 3), left=self.patch_size * (patch_id % 3),
                          height=self.patch_size,
                          width=self.patch_size)

        return center_patch, patch_crop, patch_id
