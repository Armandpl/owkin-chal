import os
import torch
import wandb
import torchvision
import torch.nn as nn
import torch.optim as optim
from utils import AverageMeter
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
)
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from histo_dataset import HistoDataset
from histo_features_dataset import HistoFeaturesDataset
from models import Resnet18Rnn
from lstm import BRNN

from set_transformer.models import SetTransformer

# Check accuracy on training & test to see how good our model
def get_accuracy(y_true, y_pred):
    assert y_true.ndim == 1 and y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    return (y_true == y_pred).sum().item() / y_true.size(0)

def get_accuracy_per_class(y_true, y_pred):
    y_pred = y_pred > 0.5
    #Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #The diagonal entries are the accuracies of each class

    return cm.diagonal()

def evaluate(loader, model):
    print("Evaluate")

    # Set model to eval
    model.eval()

    accuracy = AverageMeter()
    positive_accuracy = AverageMeter()
    negative_accuracy = AverageMeter()
    y_true = None
    y_scores = None

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device).to(torch.float32)
            y = y.to(device=device).to(torch.float32)

            scores = model(x)
            scores = torch.squeeze(scores, 2)

            y = torch.unsqueeze(y, 1)
            loss = criterion(scores, y)


            scores = torch.squeeze(scores, 1)
            y = torch.squeeze(y, 1)

            if y_true is None:
                y_true = y
                y_scores = scores
            else:
                y_true = torch.cat((y_true, y))
                y_scores = torch.cat((y_scores, scores))

            acc = get_accuracy(y, scores)
#            neg_acc, pos_acc = get_accuracy_per_class(y.cpu(), scores.cpu())

            accuracy.update(acc)
            # AUC.update(auc)
            # positive_accuracy.update(pos_acc)
            # negative_accuracy.update(neg_acc)

    
    auc = roc_auc_score(y_true.cpu(), y_scores.cpu())
    print(loss)

    wandb.log({
    "valid_acc": accuracy.avg,
#    "positive_acc": positive_accuracy.avg,
#    "negative_acc": negative_accuracy.avg,
    "valid_loss": loss.item(),
    "AUC": auc
    })

    accuracy.reset()

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # init wandb
    hyperparameters_defaults = dict(
        learning_rate = 1e-3,
        hidden_size=256,
        layers=2,
        batch_size = 32,
        num_epochs = 25,
        weight_decay = 1e-6,
        model="SetTransformer",
        aug = False,
    )

    run = wandb.init(project="owkin-chal", job_type='train', config=hyperparameters_defaults)
    config = wandb.config

    # init model
    device = torch.device("cuda")

    dataset = HistoFeaturesDataset("data/train_input/resnet_features", "data/training_output.csv")        

    train_len = int(len(dataset)*0.75)
    test_len = len(dataset)-train_len
    trainset, testset  = torch.utils.data.random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    # loaders
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # model = BRNN(2051, config.hidden_size, config.layers, 1) 
    model = SetTransformer(2051, 1, 1, num_inds=32, dim_hidden=128, num_heads=4, ln=False)

    model.to(device)
    model.train()
   

    # loss function 
    criterion = nn.BCEWithLogitsLoss()
    # criterion = neg_log_likelihood()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-08, weight_decay=config.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    evaluate(valid_loader, model)

    # Train Network
    for epoch in range(config.num_epochs):
        print("epoch: ", epoch)
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda
            data = data.to(device=device).to(torch.float32)
            targets = targets.to(device=device).to(torch.float32)
            
            # forward
            scores = model(data)
            scores = torch.squeeze(scores, 2)
            targets = torch.unsqueeze(targets, 1)
            loss = criterion(scores, targets)

            # backward
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
            "train_loss": loss.item()
            })

            # train_loss.reset()
            if (batch_idx+1)%(len(train_loader)//3) == 0:
                evaluate(valid_loader, model)

    # save model to wandb
    torch.save(model.state_dict(), 'model.pth')
    artifact = wandb.Artifact('transformer', type='model')
    artifact.add_file('model.pth')
    run.log_artifact(artifact)
