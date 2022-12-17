from skimage import io
import glob 
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def check():
    path = "/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset"
    array_images = glob.glob(os.path.join(path,'*/*/distance*.jpg'))
    counter = 0
    bad_img_list = []
    for idx, img in enumerate(array_images):
        try:
            _ = io.imread(img)
        except:
            print(img)
            bad_img_list.append(img)
            counter += 1
        if idx%10000 == 0:
            print(bad_img_list)
    with open("/home/maf4031/focus_model/src/utils/bad_img_list.npy", "wb") as f:
        np.save(f, bad_img_list)

def _check_single_img(img):
    try:
        _ = io.imread(img)
    except:
        print(img)
        return img
    return None

def check_optmized():
    path = "/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset/IDH/new_data"
    array_images = glob.glob(os.path.join(path,'*/distance*.jpg'))
    print(len(array_images))
    with ProcessPoolExecutor(max_workers=61) as executor:
        futures = executor.map(
            _check_single_img,
            array_images,
        )
    return list(filter(None, list(futures)))

if __name__ == "__main__":
    #check()
    bad_img_list = check_optmized()
    if len(bad_img_list) == 0:
        print("No corrupted files found")
    with open("/home/maf4031/focus_model/src/utils/bad_img_list_IDH.npy", "wb") as f:
        np.save(f, bad_img_list)
    #img = io.imread("/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset/Inflammation_3_4/sample_17/distance9.jpg") 