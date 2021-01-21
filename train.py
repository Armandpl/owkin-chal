import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
)
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

# Check accuracy on training & test to see how good our model
def evaluate(loader, model):
    print("Evaluate")
    # Set model to eval
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y=y/config.max_target
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y.float().unsqueeze(1))
            test_loss.update(float(loss.item())*config.max_target)

        wandb.log({
            "valid_loss": test_loss.avg
        })
        test_loss.reset()

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # init wandb
    hyperparameters_defaults = dict(
        learning_rate = 1e-3,
        batch_size = 64,
        num_epochs = 25,
        model_depth = 50,
        aug = true,
        pretrained = true
    )

    run = wandb.init(project="owkin-chal", job_type='train', config=hyperparameters_defaults)
    config = wandb.config

    # init model
    device = torch.device("cuda")
    model_name = "resnet{depth}".format(depth=config.model_depth)
    model = torchivision.models.__dict__[model_name](pretrained=config.pretrained)
    model.fc = torch.nn.linear(512, 1)
    
    model = model.to(device)

    # set transforms
    augs = [transforms.ToTensor()]

    if config.aug:
        augs.append(transforms.RandomHorizontalFlip(p=0.5)
        augs.append(transforms.RandomVerticalFlip(p=0.5)

    tfms = transforms.Compose[augs]

    

    



    

