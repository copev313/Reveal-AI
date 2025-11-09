import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomImageDataset(Dataset):
    """Dataset for loading images from a directory structure."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.img_folder = ImageFolder(root=self.root_dir)
        self.samples = self.image_folder.samples
        # ...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        # ...
        return image, target


class ImageDataModule(pl.LightningDataModule):
    """Prepares data for training and validation, applying necessary transformations."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = self.config["data"]["path"]
        self.batch_size = self.config["data"]["batch_size"]
        self.num_workers = self.config["data"]["num_workers"]
        self.validation_split = self.config["data"]["validation_split"]
        self.transforms = None

    def setup(self, stage=None):
        # ...
        pass

    def train_dataloader(self):
        # ...
        pass

    def val_dataloader(self):
        # ...
        pass
