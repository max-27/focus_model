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
from typing import List, Tuple, Callable
from src.utils.color_filter import ColorFilter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


class FocusDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        transform=None, 
        subsample: bool = False, 
        subsample_size: int = 20, 
        select_patches_random: bool = False,
        select_patches_grid: bool = False,
        patch_size: List =  [360, 256],
        image_size: List = [720, 1280],
        patch_num: int = 10,
        ) -> None:
        super().__init__()
        self.color_filter = ColorFilter()
        self.select_patches = True if select_patches_random or select_patches_grid else False
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.transform = transform
        self.array_images, self.array_labels = [], []
        if subsample:
            self._find_files_by_sample(subsample_size)
        else:
            self._find_files()

        if select_patches_random:
            self._select_patches(self._select_patches_randomly)
        elif select_patches_grid:
            self._select_patches(self._select_patches_from_grid)

        if len(self.array_images) != len(self.array_labels):
            raise ValueError("Number of images and labels do not match.")


    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        if self.select_patches:
            img_path, x, y = self.array_images[idx]
            patch = np.array(io.imread(img_path))
            patch = patch[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
            label = np.array([self.array_labels[idx] * 3.4])
        else:
            patch = np.array(io.imread(self.array_images[idx]))
            label = np.array([self.array_labels[idx] * 3.4])  # one distance unit is 3.4um
        
        if self.transform:
            patch = self.transform(patch)

        if type(patch) != torch.Tensor:
            patch = transforms.ToTensor()(patch)
        if type(label) != torch.Tensor:
            label = torch.tensor(label, dtype=torch.float)
        return patch, label

    def _find_files(self) -> None:
        self.array_images = glob.glob(os.path.join(self.data_dir,'*/*/distance*.jpg'))
        self.array_labels = [int(re.findall(r"-?\d+", path)[-1]) for path in self.array_images]
    
    def _find_files_by_sample(self, subsample_size: int = 20) -> None:
        for sample_box in next(os.walk(self.data_dir))[1]:
            samples = glob.glob(os.path.join(self.data_dir, sample_box,'sample*'))
            for sample in samples:
                images = glob.glob(os.path.join(sample,'*distance*.jpg'))
                labels = [int(re.findall(r"-?\d+", path)[-1]) for path in images]
                sampled_images, samples_labels = self._sample_images(images, labels, subsample_size)
                self.array_images.extend(sampled_images)
                self.array_labels.extend(samples_labels)
    
    def _sample_images(self, images: List, labels: List, subsample_size: int = 50) -> Tuple[List, List]:
        """Sample images with positive and negative distance values."""
        images_pos_distance = [path for path in images if int(re.findall(r"-?\d+", path)[-1]) > 0]
        images_neg_distance = [path for path in images if int(re.findall(r"-?\d+", path)[-1]) < 0]
        optimal_img = [path for path in images if int(re.findall(r"-?\d+", path)[-1]) == 0][0]

        num_pos_indices = subsample_size if subsample_size < len(images_pos_distance) else len(images_pos_distance)
        num_neg_indices = subsample_size if subsample_size < len(images_neg_distance) else len(images_neg_distance)
        pos_indices = np.random.permutation(len(images_pos_distance))[:num_pos_indices]
        neg_indices = np.random.permutation(len(images_neg_distance))[:num_neg_indices]
        pos_images = [images_pos_distance[i] for i in pos_indices]
        neg_images = [images_neg_distance[i] for i in neg_indices]

        pos_labels = [int(re.findall(r"-?\d+", path)[-1]) for path in pos_images]
        neg_labels = [int(re.findall(r"-?\d+", path)[-1]) for path in neg_images]
        return [*pos_images, *neg_images, optimal_img], [*pos_labels, *neg_labels, 0]

    def _select_patches(self, select_func: Callable) -> None:
        patch_array, label_array = [], []
        for idx, img in enumerate(self.array_images):
            patches, labels = self._select_patches_from_grid(img)
            patch_array.extend(patches)
            label_array.extend(labels)
            if idx%10000 == 0:
                print(f"Processed {idx+1} images")
        self.array_images = patch_array
        self.array_labels = label_array

    def _select_patches_randomly(self, img_path: str, patch_size: List = [360, 256], patch_num: int = 10, threshold: float = 5.) -> List:
        # select random patches randomly from each image
        patches = []
        patch_labels = []
        image = io.imread(img_path)
        h, w, _ = image.shape
        for _ in range(patch_num):
            x = np.random.randint(int(patch_size[0]/2), h - patch_size[0])
            y = np.random.randint(int(patch_size[1]/2), w - patch_size[1])
            patch = image[x-int(patch_size[0]/2):x+int(patch_size[0]/2), y-int(patch_size[1]/2):y+int(patch_size[1]/2)]
            if self.color_filter._get_sample_ratio(patch) >= threshold:
                patches.append((img_path, x, y))
                patch_labels.append(int(re.findall(r"-?\d+", img_path)[-1]))
        return patches, patch_labels

    def _select_patches_from_grid(self, img_path: str, patch_size: List = [360, 256], image_size: List = [720, 1280], threshold: float = 5.) -> List:
        # select patches from a grid from each image
        h, w = image_size
        x_steps = int(h / patch_size[0])
        y_steps = int(w / patch_size[1])
        x_coord = np.linspace(int(patch_size[0]/2), h - int(patch_size[0]/2), x_steps)
        y_coord = np.linspace(int(patch_size[1]/2), w - int(patch_size[1]/2), y_steps)
        patch_coords = [(x, y) for x in x_coord for y in y_coord]

        patches = []
        patch_labels = []
        image = io.imread(img_path)
        for x, y in patch_coords:
            patch = image[int(x-int(patch_size[0]/2)):int(x+int(patch_size[0]/2)), int(y-int(patch_size[1]/2)):int(y+int(patch_size[1]/2))]
            sample_ratio = self.color_filter._get_sample_ratio(patch)
            if sample_ratio >= threshold:
                patches.append((img_path, int(x), int(y)))
                patch_labels.append(int(re.findall(r"-?\d+", img_path)[-1]))
        if len(patches) < 5:
            curr_max_sample_ratio = 0
            curr_max_patch = None
            x_coord = int(h/2)
            y_coord = np.linspace(int(patch_size[1]), w - patch_size[1], y_steps)[:-1]
            alt_patch_coords = [(x,y) for y in y_coord]
            for x, y in alt_patch_coords:
                patch = image[int(x-int(patch_size[0]/2)):int(x+int(patch_size[0]/2)), int(y-int(patch_size[1]/2)):int(y+int(patch_size[1]/2))]
                sample_ratio = self.color_filter._get_sample_ratio(patch)
                if sample_ratio > curr_max_sample_ratio:
                    curr_max_sample_ratio = sample_ratio
                    curr_max_patch = (x, y)
                if sample_ratio >= threshold:
                    patches.append((img_path, int(x), int(y)))
                    patch_labels.append(int(re.findall(r"-?\d+", img_path)[-1]))
        if len(patches) < 5:
            patches.append((img_path, *curr_max_patch))
            patch_labels.append(int(re.findall(r"-?\d+", img_path)[-1]))
        return patches, patch_labels


if __name__ == "__main__":
    import time
    pl.seed_everything(42, workers=True)
    subsample_size = 100
    start_time = time.time()
    dataset = FocusDataset(data_dir="/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset", subsample=True, subsample_size=subsample_size, select_patches_grid=True, patch_size=[360, 256])
    print(f"Time to load dataset: {time.time() - start_time}")
    #torch.save(dataset, f"/home/maf4031/focus_model/data/datasets/dataset_subsample{subsample_size}_grid1.pt")
    torch.save(dataset, f"/home/maf4031/focus_model/data/datasets/dataset_subsample{subsample_size}_grid_complete.pt")
    