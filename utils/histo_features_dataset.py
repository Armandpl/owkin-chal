from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class HistoFeaturesDataset(Dataset):
    def __init__(self, root_dir, labels_path, test=False):
        self.root_dir = root_dir
        if not test:
            self.labels = pd.read_csv(labels_path)
        self.test = test
        self.items = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname = self.items[idx]

        features = np.load(os.path.join(self.root_dir, fname), allow_pickle=True) 
        target_size = np.zeros([1000, features.shape[1]]) 
        
        target_size[:features.shape[0],:] = features

        ID = int(fname.split("_")[1].replace(".npy",""))

        if not self.test:
            target = self.labels.loc[self.labels["ID"] == ID]["Target"]
        else:
            target = -1

        if self.test:
            return ID, target_size, int(target)
        else:
            return target_size, int(target)
