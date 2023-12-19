from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = Dataset(self.dataset_dir, variant='train')
            self.val = Dataset(self.dataset_dir, variant='test')
        if stage == 'test':
            self.test = Dataset(self.dataset_dir, variant='test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
