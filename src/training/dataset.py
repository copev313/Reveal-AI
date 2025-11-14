import os
import torch
import pytorch_lightning as pl
import numpy as np
import torchvision as tv

from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class CustomImageDataset(Dataset):
    """Dataset for loading images from a directory structure."""

    def __init__(self, root_dir: str, transforms=None):
        self.root_dir = root_dir
        self.img_folder = ImageFolder(root=self.root_dir)
        self.samples = self.img_folder.samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        img_arr = np.array(image)
        # Apply transformations (if applicable):
        if self.transforms:
            aug = self.transforms(image=img_arr)
            image = aug["image"]
        else:
            image = ToTensorV2()(image=img_arr)["image"]

        return image, target


class ImageDataModule(pl.LightningDataModule):
    """Prepares data for training and validation, applying necessary transformations."""

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.data_dir = self.config["data"]["path"]
        self.batch_size = self.config["data"]["batch_size"]
        self.img_size = self.config["data"]["image_size"]
        self.num_workers = self.config["data"].get("num_workers", 4)
        self.val_split = self.config["data"].get("validation_split", 0.2)
        self.shuffle = self.config["data"].get("shuffle", False)
        self._persist_workers = True if self.num_workers > 0 else False

        # Define transformations:
        self.training_transforms = Compose(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
                ToTensorV2(),
            ]
        )
        self.validation_transforms = Compose(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        ds = CustomImageDataset(root_dir=self.data_dir)
        # Calculate split sizes:
        _total_size = len(ds)
        _val_size = int(_total_size * self.val_split)
        _train_size = _total_size - _val_size
        # Apply random split:
        self.train_dataset, self.val_dataset = random_split(
            dataset=ds,
            lengths=[_train_size, _val_size],
        )
        # Pass transforms to datasets:
        self.train_dataset.dataset.transforms = self.training_transforms
        self.val_dataset.dataset.transforms = self.validation_transforms

    def train_dataloader(self) -> DataLoader:
        dataset = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self._persist_workers,
        )
        return dataset

    def val_dataloader(self) -> DataLoader:
        dataset = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self._persist_workers,
        )
        return dataset
