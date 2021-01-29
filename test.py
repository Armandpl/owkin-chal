import os
import torch
import wandb
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.histo_features_dataset import HistoFeaturesDataset

from models.set_transformer import SetTransformer

if __name__ == "__main__":
    # init wandb
    hyperparameters_defaults = dict(
        model="SetTransformer",
        aug = False,
    )

    run = wandb.init(project="owkin-chal", job_type='eval', config=hyperparameters_defaults)
    config = wandb.config

    artifact = run.use_artifact("transformer:v7")
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "model.pth")

    # init model
    device = torch.device("cuda")

    dataset = HistoFeaturesDataset("data/r50_features_test/", "", test=True)


    # loaders
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = SetTransformer(2051, 1, 1, num_inds=32, dim_hidden=128, num_heads=4, ln=False)
    model.load_state_dict(torch.load(model_path)) 

    model.to(device)
    model.eval()
   
    ids = []
    preds = []

    total = len(loader)
    with torch.no_grad():
        for batch_idx, (ID, x, y) in enumerate(loader):
            print(batch_idx, "/", total)
            x = x.to(device=device).to(torch.float32)
            y = y.to(device=device).to(torch.float32)

            scores = model(x)
            # scores = torch.clip(torch.squeeze(scores, 2), 0, 1)
            print(torch.squeeze(scores, 2))
            
            ids.append(int(ID))
            preds.append(float(scores))

    
    test_output = pd.DataFrame({"ID": ids, "Target": preds})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv("preds.csv")
