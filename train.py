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

# Check accuracy on training & test to see how good our model
def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def evaluate(loader, model):
    print("Evaluate")

    # Set model to eval
    model.eval()

    acc_mean = None
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device).to(torch.float32)
            y = torch.unsqueeze(y, 1)

            scores = model(x)
            loss = criterion(scores, y)

            y = torch.squeeze(y)
            scores = torch.squeeze(scores)
            acc = get_accuracy(y, scores)

            if acc_mean is None:
                acc_mean = acc
            else:
                acc_mean = (acc_mean + acc)/2
    
    wandb.log({
    "valid_acc": acc_mean,
    "valid_loss": loss.item()
    })

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # init wandb
    hyperparameters_defaults = dict(
        learning_rate = 1e-3,
        batch_size = 64,
        num_epochs = 25,
        model_depth = 50,
        aug = True,
        pretrained = True
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

    # dataset
    dataset_aug = datasets.ImageFolder("ready_to_train", transform=tfms)        
    dataset = datasets.ImageFolder("ready_to_train", transform=None)        

    train_len = int(len(dataset)*0.75)
    test_len = len(dataset)-train_len
    trainset, _  = torch.utils.data.random_split(dataset_aug, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    _ , testset  = torch.utils.data.random_split(dataset_aug, [train_len, test_len], generator=torch.Generator().manual_seed(42))

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
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-08, weight_decay=0.01)

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
