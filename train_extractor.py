import os
import torch
import wandb
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from utils.utils import AverageMeter, get_accuracy_per_class, get_accuracy

def evaluate(loader, model):
    print("Evaluate")

    # Set model to eval
    model.eval()

    accuracy = AverageMeter()
    positive_accuracy = AverageMeter()
    negative_accuracy = AverageMeter()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device).to(torch.float32)
            y = torch.unsqueeze(y, 1)

            scores = model(x)
            loss = criterion(scores, y)

            scores = torch.squeeze(scores, 1)
            y = torch.squeeze(y, 1)

            acc = get_accuracy(y, scores)
            neg_acc, pos_acc = get_accuracy_per_class(y.cpu(), scores.cpu())

            accuracy.update(acc)
            positive_accuracy.update(pos_acc)
            negative_accuracy.update(neg_acc)

    
    wandb.log({
    "valid_acc": accuracy.avg,
    "positive_acc": positive_accuracy.avg,
    "negative_acc": negative_accuracy.avg,
    "valid_loss": loss.item()
    })

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # init wandb
    hyperparameters_defaults = dict(
        learning_rate = 1e-4,
        batch_size = 64,
        num_epochs = 2,
        model_depth = 50,
        weight_decay = 0.01,
        pretrained = True,
        aug = True,
    )

    run = wandb.init(project="owkin-chal", job_type='train', config=hyperparameters_defaults)
    config = wandb.config

    # init model
    device = torch.device("cuda")
    
    model_name = "resnet{depth}".format(depth=config.model_depth)
    model = torchvision.models.__dict__[model_name](pretrained=config.pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    
    # set transforms
    augs = [transforms.ToTensor()]

    if config.aug:
        augs.append(transforms.RandomHorizontalFlip(p=0.5))
        augs.append(transforms.RandomVerticalFlip(p=0.5))

    tfms = transforms.Compose(augs)

    dataset_aug = datasets.ImageFolder("data/ready_to_train", transform=tfms)
    dataset = datasets.ImageFolder("data/ready_to_train", transform=transforms.ToTensor())

    train_len = int(dataset.__len__()*0.75)
    test_len = dataset.__len__()-train_len
    trainset, _  = torch.utils.data.random_split(dataset_aug, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    _ , testset  = torch.utils.data.random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

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

    model.to(device)
    model.train()
   
    # loss function 
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-08, weight_decay=config.weight_decay)

    evaluate(valid_loader, model)

    total = len(train_loader)
    # Train Network
    for epoch in range(config.num_epochs):
        print("epoch: ", epoch)
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda
            data = data.to(device=device)
            targets = targets.to(device=device).to(torch.float32)
            
            # forward
            scores = model(data)
            targets = torch.unsqueeze(targets, 1)
            loss = criterion(scores, targets)

            # backward
            loss.backward()

            print(batch_idx, "/", total)

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
            "train_loss": loss.item()
            })

            # train_loss.reset()
            if (batch_idx+1)%(len(train_loader)//5) == 0:
                evaluate(valid_loader, model)

    # save model to wandb
    torch.save(model.state_dict(), 'model.pth')
    artifact = wandb.Artifact('extractor', type='model')
    artifact.add_file('model.pth')
    run.log_artifact(artifact)
