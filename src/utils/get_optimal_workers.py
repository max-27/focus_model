import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# script based on this blog post: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
from time import time
import multiprocessing as mp
from src.datamodules.focus_datamodule import FocusDataModule
from src.datamodules.components.focus_dataset import FocusDataset
from torch.utils.data import DataLoader

path = "/Volumes/FOCUS/Inflammation2_665716"
batch_size = 1

optimal_time = None
optimal_num_workers = None
for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = DataLoader(FocusDataset(path, subsample=True, subsample_size=1), shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    if optimal_time is None or end-start < optimal_time:
        optimal_time = end-start
        optimal_num_workers = num_workers
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
print("Optimal num_workers is {}".format(optimal_num_workers))