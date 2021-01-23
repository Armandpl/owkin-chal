import os
import os.path
from PIL import Image
from torchvision import transforms
import torch
from collections.abc import Callable
import random
import pandas as pd
from glob import glob


class HistoDataset(torch.utils.data.Dataset):

    def __init__(self, root_path: str, transform = None):
        super(HistoDataset, self).__init__()

        self.root_path = root_path
        self.transform = transform

        self._parse_list()

    def _parse_list(self):
        self.sample_list = []

        # load labels
        labels = pd.read_csv(os.path.join(self.root_path, "training_output.csv"))
        
        # parse images
        imgs_dir = os.path.join(self.root_path, "train_input/images")
        dirs = glob(imgs_dir+"/*/")
        
        for d in dirs:
            # parse ID
            ID = int(d.split("/")[-2].split("_")[1])

            fnames = os.listdir(d)
            if len(fnames) != 1000:
                continue

            # get images paths
            imgs_paths = []
            for fname in fnames:
                imgs_paths.append(os.path.join(d, fname))
            
            # get label
            row = labels.loc[labels["ID"]==ID]
            label = int(row["Target"])
        
            self.sample_list.append((imgs_paths, label))
        

    def __getitem__(self, index):
        # load images in list
        sample = self.sample_list[index]

        images = []
        for pth in sample[0]:
            images.append(Image.open(pth))

        if self.transform is not None:
            images = [self.transform(pic) for pic in images]

        label = sample[1]
        images = torch.stack(images, 0).permute(0, 1, 2, 3)
        return images, label

    def __len__(self):
        return len(self.sample_list)

