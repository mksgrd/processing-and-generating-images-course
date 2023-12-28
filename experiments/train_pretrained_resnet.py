from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from data.datamodule import ImagewoofDatamodule
from model.resnet import ResNetForImageClassification, ResNetForVCP

if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[1]

    datamodule = ImagewoofDatamodule(
        data_dir=root_dir / 'imagewoof2',
        ratio=0.1,
        batch_size=64,
        num_workers=4
    )

    info = datamodule.info

    model = ResNetForImageClassification(
        num_classes=info.num_classes,
        id2class=info.id2class
    )

    model.feature_extractor = ResNetForVCP.load_from_checkpoint(
        root_dir / 'checkpoints' / 'epoch=35-train_f1=0.82.ckpt').feature_extractor

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=100,
        accelerator='gpu',
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'checkpoints',
                filename='{epoch}-{val_f1score_macro_avg:.2f}',
                monitor='val_f1score_macro_avg',
                mode='max',
            ),
            # EarlyStopping(
            #     monitor='val_f1score_macro_avg',
            #     mode='max',
            #     patience=25,
            # )
        ]
    )

    trainer.fit(
        model=model,
        datamodule=datamodule
    )
