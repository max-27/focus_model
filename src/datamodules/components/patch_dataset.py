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
from src.datamodules.components.focus_dataset import FocusDataset
import itertools
from concurrent.futures import ProcessPoolExecutor
from collections import ChainMap


class PatchFocusDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        transform=None, 
        subsample_size: int = 20, 
        image_size: List = [720, 1280],
        patch_size: List =  [360, 256], 
        ) -> None:
        super().__init__()
        h, w = image_size
        x_steps = int(h / patch_size[0])
        y_steps = int(w / patch_size[1])
        x_coord = np.linspace(int(patch_size[0]/2), h - int(patch_size[0]/2), x_steps)
        y_coord = np.linspace(int(patch_size[1]/2), w - int(patch_size[1]/2), y_steps)
        self.patch_coords = [(x, y) for x in x_coord for y in y_coord]
        self.patch_size = patch_size
        self.label_dict = None

        self.color_filter = ColorFilter()
        self.transform = transform
        self.data_dir = data_dir
        self.transform = transform
        self.array_images, self.array_labels = self._find_files_by_sample(subsample_size=subsample_size)
        self.patch_images, self.patch_labels = self._get_patch_directories()

    def __len__(self):
        return len(self.patch_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, x, y = self.patch_images[idx]
        img = np.array(io.imread(img_path))
        patch = img[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
        label = np.array([self.patch_labels[idx] * 3.4])  # mutliple by 3.4 to get distance in um
        return self.transform(patch), torch.from_numpy(label)

    def _find_files_by_sample(self, subsample_size: int = 20) -> List:
        array_images = []
        array_labels = []
        for sample_box in next(os.walk(self.data_dir))[1]:
            samples = glob.glob(os.path.join(self.data_dir, sample_box,'sample*'))
            for sample in samples:
                images = glob.glob(os.path.join(sample,'*distance*.jpg'))
                labels = [int(re.findall(r"-?\d+", path)[-1]) for path in images]
                sampled_images, samples_labels = self.sample_images(images, labels, subsample_size)
                array_images.extend(sampled_images)
                array_labels.extend(samples_labels)
        return array_images, array_labels
    
    def sample_images(self, images: List, labels: List, subsample_size: int = 50) -> Tuple[List, List]:
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


    def brenner_gradient(self, img: np.array) -> int:
            return np.sum((img[2:]-img[:-2])**2)

    def _get_patch_label(self, sample: str) -> Dict:
        """
        Parameters
        ----------
        sample: str
            Complete path to sample directory
        
        Returns
        -------
        image_dict: Dict
            Dictionary of sample path, optimal image index and brenner gradient value for each patch.
        """
        images = glob.glob(os.path.join(sample,'*distance*.jpg'))
        image_dict = {patch: [0, 0, 0] for patch in range(len(self.patch_coords))}
        for image in images:
            img = io.imread(image)
            for patch_idx, (x, y) in enumerate(self.patch_coords):
                patch = img[int(x-int(self.patch_size[0]/2)):int(x+int(self.patch_size[0]/2)), int(y-int(self.patch_size[1]/2)):int(y+int(self.patch_size[1]/2))]
                brenner_value = self.brenner_gradient(patch)
                if brenner_value > image_dict[patch_idx][-1]:
                    img_idx = int(image.split('distance')[-1].split('.')[0])
                    image_dict[patch_idx] = img_idx , brenner_value
        return {sample: image_dict}  

    def _get_patches(self) -> Dict:
        sample_boxes = next(os.walk(self.data_dir))[1]
        samples = list(itertools.chain.from_iterable([glob.glob(os.path.join(self.data_dir, sample_box,'sample*')) for sample_box in sample_boxes]))
        with ProcessPoolExecutor(max_workers=61) as executor:
                futures = executor.map(
                    self._get_patch_label,
                    samples,
                )
        return dict(ChainMap(*list(futures)))

    def _get_patch_directories(self) -> Dict:
        """Based on paths in array images get all patches for each sample and adjust labels of each patch."""
        patch_images = []
        patch_labels = []
        patches_dict = self._get_patches()
        for image in self.array_images:
            patch_optimal_indices = patches_dict[image.split("/distance")[0]]
            for patch_idx, (optimal_idx, _) in patch_optimal_indices.items():
                if abs(optimal_idx) < 50:
                    patch_images.append((image, self.patch_coords[patch_idx]))
                    old_idx = int(image.split('distance')[-1].split('.')[0])
                    patch_labels.append(old_idx - optimal_idx)
        return patch_images, patch_labels


if __name__ == "__main__":
    import time 
    pl.seed_everything(42, workers=True)
    start = time.time()
    dataset = PatchFocusDataset(data_dir="/Volumes/FOCUS_DATA/focus_dataset", subsample_size=100)
    print(time.time() - start)
    torch.save(dataset, "/Users/max/Desktop/Masterthesis/code/focus_model/data/test1_patch_dataset.pt")
