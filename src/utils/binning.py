import numpy as np
from PIL import Image
from pathlib import Path
import glob
import os
from tqdm import tqdm


def bin_image(image, bin_ratio):
    # Convert the input image to a numpy array
    img_arr = np.array(image)

    # Calculate the new dimensions of the binned image
    new_width = img_arr.shape[1] // bin_ratio
    new_height = img_arr.shape[0] // bin_ratio

    # Create a new numpy array for the binned image
    binned_arr = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Loop through the pixels of the binned image and average the color values of the corresponding pixels in the original image
    for y in range(new_height):
        for x in range(new_width):
            # Calculate the indices of the pixels in the original image that correspond to this binned pixel
            y_start = y * bin_ratio
            y_end = (y + 1) * bin_ratio
            x_start = x * bin_ratio
            x_end = (x + 1) * bin_ratio

            # Average the color values of the corresponding pixels in the original image
            color_sum = np.sum(img_arr[y_start:y_end, x_start:x_end], axis=(0, 1))
            color_avg = color_sum // (bin_ratio ** 2)

            # Set the color values of the binned pixel in the new array
            binned_arr[y, x] = color_avg

    # Convert the binned numpy array back to an image
    binned_img = Image.fromarray(binned_arr)

    return binned_img


import numpy as np
from PIL import Image

def bin_image_optimized(image, bin_ratio):
    # Convert the input image to a numpy array
    img_arr = np.array(image)

    # Scale the color range of the input array to 0-255
    img_arr_scaled = img_arr.astype(np.float32) / 255.0

    # Calculate the new dimensions of the binned image
    new_shape = (img_arr.shape[0] // bin_ratio, img_arr.shape[1] // bin_ratio, 3)

    # Reshape the input image to a new shape that makes binning easier
    img_reshaped = img_arr_scaled[:new_shape[0] * bin_ratio, :new_shape[1] * bin_ratio].reshape(new_shape[0], bin_ratio, new_shape[1], bin_ratio, 3)

    # Compute the mean of each bin using NumPy array slicing
    bin_means = img_reshaped.mean(axis=(1, 3)).astype(np.float64)

    # Scale the color range of the output array back to 0-255
    bin_means_scaled = (bin_means * 255.0).astype(np.uint8)

    # Convert the binned numpy array back to an image
    binned_img = Image.fromarray(bin_means_scaled)

    return binned_img


data_dir = "/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/testRawData_incoherent_sameProtocol"
save_dir = "/n/data2/hms/dbmi/kyu/lab/maf4031/incoherent_RGBchannels/test_images_binned4_sameProtocol"
Path(save_dir).mkdir(parents=True, exist_ok=True)
array_images = glob.glob(os.path.join(data_dir,'*/*.jpg'))
for img_path in tqdm(array_images):
    folder_name = img_path.split("/")[-2]
    img_name = img_path.split("/")[-1]
    Path(os.path.join(save_dir, folder_name)).mkdir(parents=True, exist_ok=True)
    image = Image.open(img_path)
    binned_img = bin_image_optimized(image, 4)
    binned_img.save(os.path.join(save_dir, folder_name, f"binned_{img_name}"))