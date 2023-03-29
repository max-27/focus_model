import os 
import re
import cv2
import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
from skimage import io
from typing import List, Tuple, Callable


def _select_patches_from_grid(img_path, threshold=50.) -> List:
    # select patches from a grid from each image


    return patches, patch_labels


data_dir = "/Users/max/Desktop/incoherent_RGBchannels/testRawData_incoherent_sameProtocol"
save_dir = "/Users/max/Desktop/incoherent_RGBchannels/test_patches_sameProtocol"
Path(save_dir).mkdir(parents=True, exist_ok=True)

image_size = (2048, 2448)
patch_size = (224, 224)
array_images, array_labels = [], []

h, w = image_size
x_steps = int(h / patch_size[0])
y_steps = int(w / patch_size[1])
x_coord = np.linspace(int(patch_size[0]/2), h - int(patch_size[0]/2), x_steps)
y_coord = np.linspace(int(patch_size[1]/2), w - int(patch_size[1]/2), y_steps)
patch_coords = [(int(x), int(y)) for x in x_coord for y in y_coord]

array_images = glob.glob(os.path.join(data_dir,'*/*.jpg'))
array_labels = [int(path.split("defocus")[-1].split(".")[0]) for path in array_images]

patches = []
patch_labels = []
for img_path in tqdm(array_images):
    folder_name = img_path.split("/")[-2]
    img_name = img_path.split("/")[-1]
    Path(os.path.join(save_dir, folder_name)).mkdir(parents=True, exist_ok=True)
    image = io.imread(img_path)
    for x, y in patch_coords:
        patch = image[int(x-int(patch_size[0]/2)):int(x+int(patch_size[0]/2)), int(y-int(patch_size[1]/2)):int(y+int(patch_size[1]/2))]
        cv2.imwrite(os.path.join(save_dir, folder_name, f"{int(x)}_{int(y)}_{img_name}.jpg"), patch)
        #patches.append((img_path, int(x), int(y)))
        #patch_labels.append(int(re.findall(r"-?\d+", img_path.split("/")[-1])[-1]))
