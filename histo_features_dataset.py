from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class HistoFeaturesDataset(Dataset):
    def __init__(self, root_dir, labels_path):
        self.root_dir = root_dir
        self.labels = pd.read_csv(labels_path)
        self.items = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname = self.items[idx]
        target_size = np.zeros([1000, 2051]) 

        features = np.load(os.path.join(self.root_dir, fname), allow_pickle=True) 
        
        target_size[:features.shape[0],:] = features

        ID = int(fname.split("_")[1].replace(".npy",""))

        target = self.labels.loc[self.labels["ID"] == ID]["Target"]
        return target_size, int(target)
