import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from utils.helpers import load_config

from training.dataset import ImageDataModule
from models.classification_model import ImageClassifier

# -----------------------------------------------------------------------------


def main(config_path: str):
    # Load config:
    config = load_config(config_path)

    # Set seed:
    pl.seed_everything(config.get("seed"))

    # Prepare data:
    data_module = ImageDataModule(config)

    # Initialize model:
    net = ImageClassifier(config)

    # Setup logging:
    logger = CSVLogger(
        save_dir=config["logging"].get("logs_dir", "logs"),
    )

    # Setup callbacks:
    callbacks = []

    # Enable learning rate monitoring hook:
    if config["training"].get("lr_monitoring"):
        monitor_lr = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(monitor_lr)

    # Enable early stopping hook:
    estop = config["training"].get("early_stopping", {})
    if estop.get("enabled"):
        early_stopping = EarlyStopping(
            monitor=estop.get("monitor", "val_loss"),
            patience=estop.get("patience", 5),
            mode=estop.get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Enable model checkpointing hook:
    checkpoint = ModelCheckpoint(
        dirpath=config["logging"].get("checkpoint_dir", "checkpoints"),
        filename="{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        monitor="val_acc",
        mode="max",
        save_last=True,
    )
    callbacks.append(checkpoint)

    # Configure training session:
    trainer = pl.Trainer(
        min_epochs=config["training"].get("min_epochs"),
        max_epochs=config["training"].get("max_epochs", -1),
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=5,
        callbacks=callbacks,
    )

    # Begin training:
    trainer.fit(
        model=net,
        datamodule=data_module,
        # Resume from an existing checkpoint:
        # ckpt_path='checkpoints/last.ckpt'
    )


if __name__ == "__main__":
    main(config_path="src/configs/example.yaml")
