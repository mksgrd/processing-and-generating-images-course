import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryRecall, BinarySpecificity


class AutoEncoderForAnomalyDetection(LightningModule):
    def __init__(self, threshold):
        super().__init__()

        self.threshold = threshold

        self.model = ConvolutionalAutoencoder()
        self.loss = nn.MSELoss(reduction='none')

        metrics = MetricCollection({
            'tpr': BinaryRecall(),
            'tnr': BinarySpecificity()
        })

        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-2)

    def training_step(self, batch, batch_idx):
        pixel_values = batch

        recon_img = self(pixel_values)
        loss = self.loss(recon_img, pixel_values).mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def common_eval_step(self, batch, metrics):
        pixel_values, labels = batch

        recon_img = self(pixel_values)
        loss = self.loss(recon_img, pixel_values).mean(axis=[1, 2, 3])
        preds = loss > self.threshold

        self.log_dict(metrics(preds, labels), prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.common_eval_step(batch, self.val_metrics)

    def test_step(self, batch, batch_idx):
        self.common_eval_step(batch, self.test_metrics)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        layer_count = 5
        in_channels = 3
        out_channels = 32

        encoder_layers = []
        decoder_layers = []

        for i in range(layer_count):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1))
            encoder_layers.append(nn.LeakyReLU())
            if i == 0:
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.LeakyReLU())
            decoder_layers.append(nn.ConvTranspose2d(out_channels, in_channels, 3, 2, 1, 1))
            in_channels = out_channels
            out_channels *= 2
        decoder_layers.reverse()

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
