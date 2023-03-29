import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pytorch_lightning as pl
from skimage import io
import os
import glob
import numpy as np
from torchvision.transforms import transforms
import re
from typing import List, Tuple, Callable
from src.utils.color_filter import ColorFilter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


class JiangDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None, 
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.array_images, self.array_labels = [], []

        self._find_files()
        if len(self.array_images) != len(self.array_labels):
            raise ValueError("Number of images and labels do not match.")

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = np.array(io.imread(self.array_images[idx]))
        label = np.array([self.array_labels[idx]])
        
        if self.transform:
            patch = self.transform(patch)

        if type(patch) != torch.Tensor:
            patch = transforms.ToTensor()(patch)
        if type(label) != torch.Tensor:
            label = torch.tensor(label, dtype=torch.float)
        return patch, label, self.array_images[idx].split('/')[-2:]

    def _find_files(self) -> None:
        self.array_images = glob.glob(os.path.join(self.data_dir,'*/*.jpg'))
        self.array_labels = [int(path.split("defocus")[-1].split(".")[0]) for path in self.array_images]


if __name__ == "__main__":
    import time
    
    pl.seed_everything(42, workers=True)
    data_dir = "/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/train_incoherent_RGBChannels"

    start_time = time.time()
    train_dataset = JiangDataset(data_dir=data_dir)
    print(len(train_dataset))
    print(f"Time to load dataset: {time.time() - start_time}")
    torch.save(train_dataset, f"/home/maf4031/focus_model/data/jiang_datasets/dataset_patch_train.pt")

    data_dir = "/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/test_patches_binned4_sameProtocol"
    start_time = time.time()
    test_dataset = JiangDataset(data_dir=data_dir)
    print(f"Time to load dataset: {time.time() - start_time}")
    torch.save(test_dataset, f"/home/maf4031/focus_model/data/jiang_datasets/dataset_patch_binned4_test_same.pt")

    #data_dir = "/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/testRawData_incoherent_diffProtocol"
    #start_time = time.time()
    #test_dataset = PatchNewDataset(data_dir=data_dir)
    #print(f"Time to load dataset: {time.time() - start_time}")
    #torch.save(test_dataset, f"/home/maf4031/focus_model/data/jiang_datasets/dataset_patch_test_diff.pt")
