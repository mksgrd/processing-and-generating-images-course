from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataset import ImagewoofDataset, ImagewoofVCPDataset


class ImagewoofDatamodule(LightningDataModule):
    def __init__(self, data_dir, ratio, batch_size, num_workers):
        super().__init__()

        self.data_dir = data_dir
        self.ratio = ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.info = ImagewoofDataset.get_info()

        self.train = None
        self.val = None

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = ImagewoofDataset(data_dir=self.data_dir, ratio=self.ratio, train=True)
            self.val = ImagewoofDataset(data_dir=self.data_dir, ratio=self.ratio, train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )


class ImagewoofVCPDatamodule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.info = ImagewoofDataset.get_info()

        self.train = None

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = ImagewoofVCPDataset(data_dir=self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True
        )
