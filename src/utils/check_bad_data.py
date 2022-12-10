from skimage import io
import glob 
import os


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
    print(bad_img_list)


if __name__ == "__main__":
    check()
    #img = io.imread("/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset/Inflammation_3_4/sample_17/distance9.jpg") 