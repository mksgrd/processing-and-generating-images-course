import statistics

import albumentations
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    mean = (0.4237, 0.5344, 0.4620)
    std = (0.0472, 0.0526, 0.0489)
    resize_height = 32
    resize_width = 48
    crop_height = 32
    crop_width = 32

    @staticmethod
    def get_sorted_file_paths(dir_path):
        return sorted((path for path in dir_path.iterdir() if path.is_file()), key=lambda path: path.name)

    def __init__(self, dataset_dir, variant):
        self.variant = variant
        if self.variant == 'train':
            self.image_paths = self.get_sorted_file_paths(dataset_dir / 'train')
        elif self.variant == 'anomaly':
            self.image_paths = self.get_sorted_file_paths(dataset_dir / 'proliv')
        elif self.variant == 'test':
            self.image_paths = []
            self.image_labels = []
            with open(dataset_dir / 'test' / 'test_annotation.txt', 'r') as ann_file:
                for ann in ann_file.readlines():
                    file_name, label = ann.split()
                    self.image_paths.append(dataset_dir / 'test' / 'imgs' / file_name)
                    self.image_labels.append(int(label))

        self.train_transforms = albumentations.Compose([
            albumentations.Resize(height=self.resize_height, width=self.resize_width),
            albumentations.CenterCrop(height=self.crop_height, width=self.crop_width),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.RandomBrightnessContrast(),
            albumentations.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        self.transforms = albumentations.Compose([
            albumentations.Resize(height=self.resize_height, width=self.resize_width),
            albumentations.CenterCrop(height=self.crop_height, width=self.crop_width),
            albumentations.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pil_image = Image.open(self.image_paths[index]).convert('RGB')

        if self.variant == 'train':
            transforms = self.train_transforms
        else:
            transforms = self.transforms

        transformed_image = transforms(image=np.array(pil_image))['image']

        if self.variant != 'test':
            return transformed_image
        return transformed_image, self.image_labels[index]

    def calc_mean_std(self):
        transforms = albumentations.Compose([
            albumentations.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ])
        pixel_count = 0
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        for image_path in self.image_paths:
            img = Image.open(image_path).convert('RGB')
            pixel_count += img.width * img.height
            img = transforms(image=np.array(img))['image']
            psum += img.sum(axis=[1, 2])
            psum_sq += (img ** 2).sum(axis=[1, 2])
        mean = psum / pixel_count
        var = (psum_sq / pixel_count) - (mean ** 2)
        std = torch.sqrt(var)
        return mean, std

    def calc_statistics(self):
        widths = []
        heights = []
        for image_path in self.image_paths:
            img = Image.open(image_path).convert('RGB')
            widths.append(img.width)
            heights.append(img.height)
        width_deciles = statistics.quantiles(widths, n=10)
        height_deciles = statistics.quantiles(heights, n=10)
        return width_deciles, height_deciles
