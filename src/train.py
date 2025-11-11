import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
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
    # ...

    # Initialize model:
    # net = ImageClassifier(config)

    # Setup logging:
    # logger_type = config["logging"].get("type", "").lower()
    logger = CSVLogger(
        save_dir=config["logging"].get("logs_dir", "logs"),
    )

    # Setup callbacks:
    callbacks = []
    # TODO: Handle checkpoints...

    # Configure training session:
    trainer = pl.Trainer(
        min_epochs=config["training"].get("min_epochs"),
        max_epochs=config["training"].get("max_epochs"),
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
    # trainer.fit(
    #     model=net,
    #     datamodule=...,
    #     # Resume from an existing checkpoint:
    #     # ckpt_path=...,
    # )


if __name__ == "__main__":
    main(config_path="src/configs/example.yaml")
