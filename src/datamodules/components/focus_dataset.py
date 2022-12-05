import torch
from torch.utils.data import Dataset
from skimage import io
import os
import glob
import numpy as np
from torchvision.transforms import transforms
import re


class FocusDataset(Dataset):
    def __init__(self, data_dir: str = "data/", transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.array_images, self.array_labels = self.find_files(data_dir)

    def __len__(self):
        return len(self.array_labels)

    def __getitem__(self, idx):
        patch = np.array(io.imread(self.array_images[idx]))
        label = np.array([self.array_labels[idx] * 3.4])  # one distance value is 3.4um
        
        if self.transform:
            patch = self.transform(image = patch)['image']

        patch = transforms.ToTensor()(patch)
        label = torch.tensor(label, dtype=torch.float)
        return patch, label

    def find_files(self, path):
        images = glob.glob(os.path.join(path,'*distance*.jpg')) #TODO add / to distance path
        labels = [int(re.findall(r"-?\d+", path)[-1]) for path in images]
        return images, labels


if __name__ == "__main__":
    dataset = FocusDataset(data_dir="/Users/max/Desktop/test_data")
    x, y = dataset[0]