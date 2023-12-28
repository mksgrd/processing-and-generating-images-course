import torch
import torch.nn as nn

from lightning import LightningModule
from torch.optim import AdamW
from torchmetrics import F1Score, Precision, Recall, MetricCollection


class ResNetForImageClassification(LightningModule):
    def __init__(self, num_classes, id2class):
        super().__init__()

        self.id2class = id2class

        block = ResidualBlock
        self.feature_extractor = ResNetFeatureExtractor(block, layers=[3, 4, 6, 3])
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.loss = nn.CrossEntropyLoss()

        metrics = MetricCollection({
            'precision': Precision(task='multiclass', num_classes=num_classes, average='none'),
            'recall': Recall(task='multiclass', num_classes=num_classes, average='none'),
            'f1score': F1Score(task='multiclass', num_classes=num_classes, average='none')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)

    def common_step(self, batch, metrics, log_prefix):
        pixel_values, class_id = batch
        outputs = self(pixel_values)
        loss = self.loss(outputs, class_id)
        preds = outputs.argmax(-1)

        metrics.update(preds, class_id)

        self.log(f'{log_prefix}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, self.train_metrics, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, self.val_metrics, log_prefix='val')

    def predict_step(self, batch, batch_idx):
        pixel_values = batch
        outputs = self(pixel_values)
        preds = outputs.argmax(-1)
        return [self.id2class[pred] for pred in list(preds)]

    def on_common_epoch_end(self, metrics):
        for metric_key, metric_values in metrics.compute().items():
            self.log(f'{metric_key}_macro_avg', metric_values.mean())
            for class_id, metric_value in enumerate(metric_values):
                self.log(f'{metric_key}_{self.id2class[class_id]}', metric_value)

        metrics.reset()

    def on_train_epoch_end(self):
        self.on_common_epoch_end(self.train_metrics)

    def on_validation_epoch_end(self):
        self.on_common_epoch_end(self.val_metrics)


class ResNetForVCP(LightningModule):
    def __init__(self):
        super().__init__()

        block = ResidualBlock
        self.feature_extractor = ResNetFeatureExtractor(block, layers=[3, 4, 6, 3])
        self.fc1 = nn.Linear(2 * 512 * block.expansion, 2 * 512 * block.expansion)
        self.bn1 = nn.BatchNorm1d(2 * 512 * block.expansion)
        self.fc2 = nn.Linear(2 * 512 * block.expansion, 512 * block.expansion)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc3 = nn.Linear(512 * block.expansion, 9)
        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()
        self.train_f1 = F1Score(task='multiclass', num_classes=9, average='macro')

    def forward(self, patch1, patch2):
        x1 = self.feature_extractor(patch1)
        x2 = self.feature_extractor(patch2)
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        center_patch, patch_crop, patch_id = batch
        outputs = self(center_patch, patch_crop)
        preds = outputs.argmax(-1)
        loss = self.loss(outputs, patch_id)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1(preds, patch_id), prog_bar=True, on_step=False, on_epoch=True)

        return loss


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
