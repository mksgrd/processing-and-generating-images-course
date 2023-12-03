from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from data.datamodule import DataModule
from model.resnet import ResNetForImageClassification

if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[1]
    data_dir = root_dir / 'imagewoof2'

    data_module = DataModule(
        predict=False,
        data_dir=data_dir,
        batch_size=48,
        num_workers=4
    )

    data_info = data_module.info

    model = ResNetForImageClassification(
        num_classes=data_info.num_classes,
        id2class=data_info.id2class
    )

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=-1,
        accelerator='gpu',
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'checkpoints',
                filename='{epoch}-{val_f1score_macro_avg:.2f}',
                monitor='val_f1score_macro_avg',
                mode='max',
                save_last=True,
                save_top_k=3
            ),
            EarlyStopping(
                monitor='val_f1score_macro_avg',
                mode='max',
                patience=30,
            )
        ]
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path='last'
    )
