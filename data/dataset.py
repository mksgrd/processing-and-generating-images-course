from dataclasses import dataclass

import albumentations
import numpy as np
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as TorchDataset


@dataclass
class DatasetInfo:
    num_classes: int
    class2id: dict[str, int]
    id2class: list[str]
    label2class = {
        'n02086240': 'Shih-Tzu',
        'n02087394': 'Rhodesian_ridgeback',
        'n02088364': 'Beagle',
        'n02089973': 'English_foxhound',
        'n02093754': 'Australian_terrier',
        'n02096294': 'Border_terrier',
        'n02099601': 'Golden_retriever',
        'n02105641': 'Old_English_sheepdog',
        'n02111889': 'Samoyed',
        'n02115641': 'Dingo'
    }


class Dataset(TorchDataset):
    @staticmethod
    def get_info(data_dir):
        return Dataset(data_dir, train=True).info

    @staticmethod
    def get_default_info():
        return DatasetInfo(
            num_classes=10,
            class2id={
                'Shih-Tzu': 0,
                'Rhodesian_ridgeback': 1,
                'Beagle': 2,
                'English_foxhound': 3,
                'Australian_terrier': 4,
                'Border_terrier': 5,
                'Golden_retriever': 6,
                'Old_English_sheepdog': 7,
                'Samoyed': 8,
                'Dingo': 9
            },
            id2class=['Shih-Tzu', 'Rhodesian_ridgeback', 'Beagle', 'English_foxhound', 'Australian_terrier',
                      'Border_terrier', 'Golden_retriever', 'Old_English_sheepdog', 'Samoyed', 'Dingo']
        )

    @staticmethod
    def get_train_transforms():
        return albumentations.Compose([
            albumentations.RandomResizedCrop(height=256, width=256, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            albumentations.HorizontalFlip(),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms():
        return albumentations.Compose([
            albumentations.Resize(height=256, width=256),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __init__(self, data_dir, train):
        self.train = train

        class_dirs = sorted((
            class_dir for class_dir in (data_dir / ('train' if self.train else 'val')).iterdir() if class_dir.is_dir()
        ), key=lambda class_dir: class_dir.name)

        num_classes = 0
        class2id = {}
        id2class = []

        self.image_paths = []
        self.class_ids = []

        for class_dir in class_dirs:
            class_name = DatasetInfo.label2class[class_dir.name]
            if class_name not in class2id:
                class2id[class_name] = num_classes
                id2class.append(class_name)
                num_classes += 1

            class_id = num_classes - 1

            image_paths = sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                 key=lambda image_path: image_path.name)

            for image_path in image_paths:
                self.image_paths.append(image_path)
                self.class_ids.append(class_id)

        self.info = DatasetInfo(num_classes, class2id, id2class)

        self.train_transforms = self.get_train_transforms()
        self.val_transforms = self.get_val_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pil_image = ImageOps.exif_transpose(Image.open(self.image_paths[index]).convert('RGB'))

        if self.train:
            transforms = self.train_transforms
        else:
            transforms = self.val_transforms

        transformed_image = transforms(image=np.array(pil_image))['image']

        return transformed_image, self.class_ids[index]


class PredictDataset(TorchDataset):
    def __init__(self, data_dir):
        self.image_paths = sorted((image_path for image_path in data_dir.iterdir() if image_path.is_file()),
                                  key=lambda image_path: image_path.name)

        self.transforms = Dataset.get_val_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pil_image = ImageOps.exif_transpose(Image.open(self.image_paths[index]).convert('RGB'))
        transformed_image = self.transforms(image=np.array(pil_image))['image']
        return transformed_image
