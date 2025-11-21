import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.functional import accuracy


class ImageClassifier(pl.LightningModule):
    """Base image classification model using PyTorch Lightning."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = float(self.config["training"].get("learning_rate", 1e-4))
        self.opt_name = self.config["training"].get("optimizer_type", "adamw").lower()
        self.wt_decay = float(self.config["training"].get("weight_decay", 0))
        self.sched_name = self.config["training"].get("scheduler_type", "").lower()
        self.loss_func = nn.CrossEntropyLoss()

        self.num_classes = self.config["model"].get("num_classes", 0)
        self.backbone = timm.create_model(
            model_name=self.config["model"]["name"],
            pretrained=self.config["model"].get("pretrained", False),
            num_classes=self.num_classes,
        )

    def forward(self, x):
        out = self.backbone(x)
        return out

    def _calc_metrics(self, preds, labels):
        """Calculate accuracy and F1 score."""
        n_classes = self.num_classes
        task_type = "multiclass" if n_classes > 2 else "binary"
        acc = accuracy(preds, labels, task=task_type, num_classes=n_classes)
        return acc

    def _common_step(self, batch, batch_idx, stage: str):
        images, labels = batch
        # Forward pass:
        logits = self(images)
        # Calculate loss:
        loss = self.loss_func(logits, labels)
        # Highest prob. class wins:
        preds = torch.argmax(logits, dim=1)
        # Calculate metrics:
        acc = self._calc_metrics(preds, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        # Select optimizer by name:
        if self.opt_name == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.wt_decay,
            )
        elif self.opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.wt_decay,
            )
        elif self.opt_name == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.wt_decay,
                momentum=0.9,
            )
        else:
            raise NotImplementedError(f"Optimizer '{self.opt_name}' not implemented.")

        # Setup scheduler (if specified):
        if not self.sched_name:
            return optimizer

        if self.sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.config["training"].get("step_size", 10),
                gamma=self.config["training"].get("gamma", 0.1),
            )
        elif self.sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=self.config["training"].get("factor", 0.1),
                patience=self.config["training"].get("patience", 5),
                threshold=self.config["training"].get("threshold", 1e-4),
                min_lr=self.config["training"].get("min_lr", 0),
            )
        elif self.sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.config["training"].get("t_max", 50),
            )
        else:
            raise NotImplementedError(f"Scheduler '{self.sched_name}' not implemented.")

        # REF: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.sched_name == "plateau" else None,
                "interval": "epoch",
                "frequency": 1,
            },
        }
