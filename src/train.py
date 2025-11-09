import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from utils.helpers import load_config


def main(config_path: str):
    # Load config:
    config = load_config(config_path)

    # Set seed:
    pl.seed_everything(config.get("seed"))

    # ...

    # Configure training session:
    trainer = pl.Trainer(
        min_epochs=config["training"]["min_epochs"],
        max_epochs=config["training"]["max_epochs"],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        devices="auto",
        accelerator="auto",
        # logger=...,
        # log_every_n_steps=...,
        # callbacks=[],
    )

    # Begin training:
    # trainer.fit(
    #     model=...,
    #     train_dataloaders=...,
    #     val_dataloaders=...,
    #     datamodule=...,
    #     ckpt_path=...,
    # )


if __name__ == "__main__":
    main(config_path="src/configs/example.yaml")
