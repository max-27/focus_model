import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.datamodules.components.focus_dataset import FocusDataset


class FocusDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        splits: Tuple[float, float] = [0.8, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        self.transforms = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = FocusDataset(self.hparams.data_dir, transform=self.transforms)
            self.data_train = dataset
            self.data_val = dataset
            self.data_test = dataset

    # def setup(self, stage: Optional[str] = None) -> None:
    #     if not self.data_train and not self.data_val and not self.data_test:
    #         dataset = FocusDataset(self.hparams.data_dir, transform=self.transforms)
    #         len_dataset = len(dataset)
    #         train_size = int(len_dataset * self.hparams.splits[0])
    #         val_size = int(len_dataset * self.hparams.splits[1])
    #         test_size = len_dataset - train_size - val_size
    #         self.data_train, self.data_val, self.data_test = random_split(
    #             dataset=dataset,
    #             lengths=[train_size, val_size, test_size],
    #             generator=torch.Generator().manual_seed(42),
    #         )

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
    datamodule = FocusDataModule()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    a = 1