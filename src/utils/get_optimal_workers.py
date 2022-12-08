# script based on this blog post: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
from time import time
import multiprocessing as mp
from src.datamodules.focus_datamodule import FocusDataModule
from torch.utils.data import DataLoader

path = "/n/data2/hms/dbmi/kyu/lab/maf4031/focus_dataset/Inflammation2_665716"
batch_size = 32

optimal_time = None
optimal_num_workers = None
for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = DataLoader(FocusDataModule(path), shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    if end-start < optimal_time or optimal_time is None:
        optimal_time = end-start
        optimal_num_workers = num_workers
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
print("Optimal num_workers is {}".format(optimal_num_workers))