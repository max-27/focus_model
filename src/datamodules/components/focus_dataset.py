import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import torch
from torch.utils.data import Dataset
from skimage import io
import os
import glob
import numpy as np
from torchvision.transforms import transforms
import re
from typing import List, Tuple


class FocusDataset(Dataset):
    def __init__(self, data_dir: str = "data/", transform=None, subsample: bool = False, subsample_size: int = 20):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.array_images, self.array_labels = [], []
        if subsample:
            self.find_files_by_sample(subsample_size)
        else:
            self.find_files()

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = np.array(io.imread(self.array_images[idx]))
        label = np.array([self.array_labels[idx] * 3.4])  # one distance unit is 3.4um
        
        if self.transform:
            patch = self.transform(patch)

        if type(patch) != torch.Tensor:
            patch = transforms.ToTensor()(patch)
        if type(label) != torch.Tensor:
            label = torch.tensor(label, dtype=torch.float)
        return patch, label

    def find_files(self) -> None:
        self.array_images = glob.glob(os.path.join(self.data_dir,'*distance*.jpg')) #TODO add / to distance path
        self.array_labels = [int(re.findall(r"-?\d+", path)[-1]) for path in self.array_images]
    
    def find_files_by_sample(self, subsample_size: int = 20) -> None:
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


if __name__ == "__main__":
    dataset = FocusDataset(data_dir="/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset", subsample=True, subsample_size=20)
    x, y = dataset[0]
    a = 1