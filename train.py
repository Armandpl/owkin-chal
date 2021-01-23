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
from sklearn.metrics import confusion_matrix
import numpy as np
from histo_dataset import HistoDataset
from models import Resnet18Rnn

def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

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
        learning_rate = 1e-3,
        batch_size = 5,
        minibatch = 5,
        num_epochs = 6,
        aug = True,
    )

    run = wandb.init(project="owkin-chal", job_type='train', config=hyperparameters_defaults)
    config = wandb.config

    # init model
    device = torch.device("cuda")
    # model = torchvision.models.__dict__['resnet18'](pretrained=True)

    # state = torch.load('_ckpt_epoch_9.ckpt', map_location='cuda:0')

    # state_dict = state['state_dict']
    # for key in list(state_dict.keys()):
    #   state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    # model = load_model_weights(model, state_dict)

    #model_name = "resnet{depth}".format(depth=config.model_depth)
    #model = torchvision.models.__dict__[model_name](pretrained=config.pretrained)
    # model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = Resnet18Rnn(rnn_hidden_size=500)
    artifact = run.use_artifact("model:v7")
    artifact_dir = artifact.download()
    model.load_state_dict_for_base(os.path.join(artifact_dir, "model.pth"))
    model.freeze_base_model()

    
    # set transforms
    augs = [transforms.ToTensor()]

    if config.aug:
        augs.append(transforms.RandomHorizontalFlip(p=0.5))
        augs.append(transforms.RandomVerticalFlip(p=0.5))

    tfms = transforms.Compose(augs)

    # dataset
    # dataset_aug = datasets.ImageFolder("ready_to_train", transform=tfms)        
    # dataset = datasets.ImageFolder("ready_to_train", transform=None)        

    dataset_aug = HistoDataset("data", transform=tfms)        
    dataset = HistoDataset("data", transform=None)        

    train_len = int(dataset.__len__()*0.75)
    test_len = dataset.__len__()-train_len
    trainset, _  = torch.utils.data.random_split(dataset_aug, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    _ , testset  = torch.utils.data.random_split(dataset_aug, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    # loaders
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.minibatch,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.minibatch,
        shuffle=False
    )

    model.to(device)
    model.train()
   
    # loss function 
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-08, weight_decay=0.01)

    evaluate(valid_loader, model)

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
            # train_loss.update(loss.item())

            # backward
            loss.backward()

            print(batch_idx)

            if (batch_idx+1)%config.minibatch == 0 or config.batch_size == config.minibatch:
                print("optimizer step")
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
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.pth')
    run.log_artifact(artifact)
