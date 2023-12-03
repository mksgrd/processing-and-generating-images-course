from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataset import Dataset, PredictDataset


class DataModule(LightningDataModule):
    def __init__(self, predict, data_dir, batch_size, num_workers):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if predict:
            self.info = Dataset.get_default_info()
        else:
            self.info = Dataset.get_info(self.data_dir)

        self.train = None
        self.val = None
        self.predict = None

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = Dataset(data_dir=self.data_dir, train=True)
            self.val = Dataset(data_dir=self.data_dir, train=False)

        if stage == 'predict':
            self.predict = PredictDataset(self.data_dir)

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

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )
