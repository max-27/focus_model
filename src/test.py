import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os 
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import pytorch_lightning as pl
from skimage import io
import numpy as np
from torchvision.transforms import transforms
import re
from typing import List, Tuple, Callable
from src.utils.color_filter import ColorFilter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchvision.transforms.functional import InterpolationMode
from typing import Dict, List, Optional, Tuple, Union, Any
from src.utils.color_filter import ColorFilter


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None, 
        ) -> None:
        super().__init__()
        self.color_filter = ColorFilter()
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = (2048, 2448)
        self.patch_size = (224, 224)
        self.array_images, self.array_labels = [], []

        h, w = self.image_size
        x_steps = int(h / self.patch_size[0])
        y_steps = int(w / self.patch_size[1])
        x_coord = np.linspace(int(self.patch_size[0]/2), h - int(self.patch_size[0]/2), x_steps)
        y_coord = np.linspace(int(self.patch_size[1]/2), w - int(self.patch_size[1]/2), y_steps)
        self.patch_coords = [(x, y) for x in x_coord for y in y_coord]

        self._find_files()
        self._select_patches()
        if len(self.array_images) != len(self.array_labels):
            raise ValueError("Number of images and labels do not match.")

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        img_path, x, y = self.array_images[idx]
        patch = np.array(io.imread(img_path))
        patch = patch[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
        label = np.array([self.array_labels[idx]])
        
        if self.transform:
            patch = self.transform(patch)

        if type(patch) != torch.Tensor:
            patch = transforms.ToTensor()(patch)
        if type(label) != torch.Tensor:
            label = torch.tensor(label, dtype=torch.float)
        return patch, label, self.array_images[idx][0].split('/')[-2:]

    def _find_files(self) -> None:
        self.array_images = glob.glob(os.path.join(self.data_dir,'*/*.jpg'))
        self.array_labels = [int(path.split("defocus")[-1].split(".")[0]) for path in self.array_images]

    def _select_patches(self) -> None:
        patch_array, label_array = [], []
        for idx, img in enumerate(tqdm(self.array_images)):
            patches, labels = self._select_patches_from_grid(img)
            patch_array.extend(patches)
            label_array.extend(labels)
        self.array_images = patch_array
        self.array_labels = label_array

    def _select_patches_from_grid(self, img_path, threshold=50.) -> List:
        # select patches from a grid from each image

        patches = []
        patch_labels = []
        image = io.imread(img_path)
        for x, y in self.patch_coords:
            patch = image[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
            sample_ratio = self.color_filter._get_sample_ratio(patch)
            if sample_ratio >= threshold:
                patches.append((img_path, int(x), int(y)))
                patch_labels.append(int(re.findall(r"-?\d+", img_path.split("/")[-1])[-1]))
        return patches, patch_labels


class TestDatamodule(LightningDataModule):
    def __init__(self, test_dataset) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_test = test_dataset

    def setup(self, stage: str = None) -> None:
        pass

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    from src.models.components.ynet_spectral import YNet_spectral
    from src.models.components.ynet_spatial import YNet_spatial
    from src.models.components.ynet_mixed import YNet_mixed

    device = "cpu"
    load_existing_dataset = False
    protocol = "same" # "diff"
    data_path = f"/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/testRawData_incoherent_{protocol}Protocol"  
    ckpt_path = "/home/maf4031/focus_model/logs/wandb_sweep/runs/2023-03-06_10-35-54/checkpoints/epoch_147.ckpt"

    if not load_existing_dataset:
        dataset = TestDataset(data_path)
        torch.save(dataset, f"/home/maf4031/focus_model/data/test/{protocol}_test_dataset.pt")
    else:
        dataset = torch.load(f"/home/maf4031/focus_model/data/test/{protocol}_test_dataset.pt")
    print(len(dataset))
    dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])
    model = YNet_spectral()
    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
    state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    results_dict = {}
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch, label, patch_ids = dataset[i]
            patch_id = patch_ids[0] + '_' + patch_ids[1]
            patch = patch.unsqueeze(0).to(device)
            output = model(patch)
            if patch_id not in results_dict:
                results_dict[patch_id] = []
            results_dict[patch_id].append(output.item())
    errors = []
    for key in results_dict:
        label = float(key.split('_defocus')[-1].split(".")[0])
        mean_pred = np.mean(results_dict[key])
        errors.append(abs(mean_pred - label))
    Path(f"/home/maf4031/focus_model/output/test/{ckpt_path.split('/')[-3]}").mkdir(exist_ok=True, parents=True)
    np.save(f"/home/maf4031/focus_model/output/test/{ckpt_path.split('/')[-3]}/{protocol}_test_erros.npy", results_dict)
    print(f"Mean absolute error: {np.mean(errors)}")
    print(f"Standard deviation: {np.std(errors)}")
