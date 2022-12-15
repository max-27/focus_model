import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from skimage import io
import os
import glob
import numpy as np
from torchvision.transforms import transforms
import re
from typing import List, Tuple, Dict
from src.utils.color_filter import ColorFilter


class FocusDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        transform=None, 
        subsample: bool = False, 
        subsample_size: int = 20, 
        image_size: List = [720, 1280],
        patch_size: List =  [360, 256], 
        patch_num: int = 10,
        ) -> None:
        super().__init__()
        self.color_filter = ColorFilter()
        self.data_dir = data_dir
        self.transform = transform

        h, w = image_size
        x_steps = int(h / patch_size[0])
        y_steps = int(w / patch_size[1])
        x_coord = np.linspace(int(patch_size[0]/2), h - int(patch_size[0]/2), x_steps)
        y_coord = np.linspace(int(patch_size[1]/2), w - int(patch_size[1]/2), y_steps)
        self.patch_coords = [(x, y) for x in x_coord for y in y_coord]
        self.patch_size = patch_size
        self.label_dict = None

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(index)

    def brenner_gradient(self, img: np.array) -> int:
            return np.sum((img[2:]-img[:-2])**2)

    def get_directories(self) -> Dict:
        sample_box_dict = {sample_box: {} for sample_box in next(os.walk(self.data_dir))[1]}
        for sample_box in next(os.walk(self.data_dir))[1]:
            samples = glob.glob(os.path.join(self.data_dir, sample_box,'sample*'))
            sample_dict = {sample: {} for sample in samples}
            for sample in samples:
                images = glob.glob(os.path.join(sample,'*distance*.jpg'))
                image_dict = {patch: [0, 0, 0] for patch in range(len(self.patch_coords))}
                for image in images:
                    img = io.imread(image)
                    for patch_idx, (x, y) in enumerate(self.patch_coords):
                        patch = img[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
                        brenner_value = self.brenner_gradient(patch)
                        if brenner_value > image_dict[patch_idx][-1]:
                            image_dict[patch_idx] = x, y, brenner_value
                sample_dict[sample] = image_dict
            sample_box_dict[sample_box] = sample_dict
            print(sample_box)
        self.label_dict = sample_box_dict
        return sample_box_dict

if __name__ == "__main__":
    import time 
    pl.seed_everything(42, workers=True)
    dataset = FocusDataset(data_dir="/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset")
    start = time.time()
    dict_out = dataset.get_directories()
    total_time = time.time() - start
    torch.save(dataset, f"/home/maf4031/focus_model/data/test_patch_dataset_{total_time}.pt")
    a = 1

