from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from data.datamodule import ImagewoofVCPDatamodule
from model.resnet import ResNetForVCP

if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[1]

    datamodule = ImagewoofVCPDatamodule(
        data_dir=root_dir / 'imagewoof2',
        batch_size=64,
        num_workers=4
    )

    model = ResNetForVCP()

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=-1,
        accelerator='gpu',
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'checkpoints',
                filename='{epoch}-{train_f1:.2f}',
                monitor='train_f1',
                mode='max',
                save_last=True,
            ),
            EarlyStopping(
                monitor='train_f1',
                mode='max',
                patience=50,
            )
        ]
    )

    trainer.fit(
        model=model,
        datamodule=datamodule
    )
