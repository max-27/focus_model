import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from typing import Any, Dict, Optional, Tuple, List
import torch
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from src.datamodules.components.focus_dataset import FocusDataset


class Transformation:
    def __init__(
        self,
        params: Dict[str, Any],
    ):
        transformation_list = [transforms.ToTensor()]
        if params.resize_img:
            #h, w = [720, 1280]
            #w_scaled = int(w * 0.3)
            #h_scaled = int(h * 0.3)
            w_scaled, h_scaled = params.scaled_img_size
            transformation_list.append(transforms.Resize((h_scaled, w_scaled), interpolation=InterpolationMode.BILINEAR))
        if params.horizontal_flip:
            transformation_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if params.vertical_flip:
            transformation_list.append(transforms.RandomVerticalFlip(p=0.5))
        if params.rotation:
            transformation_list.append(transforms.RandomRotation(degrees=params.rotation_degrees))
        if params.perspective:
            transformation_list.append(transforms.RandomPerspective(*params.perspective_parameters))
        if params.color_jitter:
            transformation_list.append(transforms.ColorJitter(*params.color_jitter_parameters))
        if params.random_erasing:
            transformation_list.append(transforms.RandomErasing(p=1.,scale=(0.02, 0.1)))

        if params.normalize:
            transformation_list.append(transforms.Normalize((0), (1)))
        self.transforms = transforms.Compose([*transformation_list])
    
    def __call__(self, x):
        return self.transforms(x)


class FocusDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        dataset_dir: str = "",
        splits: Tuple[float, float] = [0.8, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        subsample: bool = False,
        subsample_size: int = 50,
        image_size: List = [1280, 720],
        resize_scaling_factor: float = 0.2,
        select_patches_grid: bool = False,
        select_patches_random: bool = False,
        transformations: Optional[Transformation] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        w, h = image_size
        w_scaled = int(w * resize_scaling_factor)
        h_scaled = int(h * resize_scaling_factor)

        self.transforms = transformations
        if self.transforms is None:
            self.transforms = self.hparams.transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            if os.path.exists(self.hparams.dataset_dir):
                dataset = torch.load(self.hparams.dataset_dir)
                dataset.transform = self.transforms
            else:
                dataset = FocusDataset(self.hparams.data_dir, transform=self.transforms, subsample=self.hparams.subsample, subsample_size=self.hparams.subsample_size, select_patches_grid=self.hparams.select_patches_grid)
            len_dataset = len(dataset)
            train_size = int(len_dataset * self.hparams.splits[0])
            val_size = int(len_dataset * self.hparams.splits[1])
            test_size = len_dataset - train_size - val_size
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass


if __name__ == "__main__":
    datamodule = FocusDataModule(
        data_dir="/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset",
        dataset_dir="/home/maf4031/focus_model/data/datasets/dataset_subsample100_grid.pt",
        subsample=True,
        )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    a = 1