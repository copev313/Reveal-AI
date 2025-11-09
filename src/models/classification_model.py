import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.functional import accuracy, f1_score


class ImageClassifier(pl.LightningModule):
    """Base image classification model using PyTorch Lightning."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = timm.create_model(
            model_name=self.config["model"]["name"],
            pretrained=self.config["model"]["pretrained"],
            num_classes=self.config["model"]["num_classes"],
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)
        # self.log(...)
        # ...
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)
        # self.log(...)
        # ...
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer
