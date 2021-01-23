import os
import pandas as pd
from glob import glob
from PIL import Image
import torch
import torchvision
from torch.utils.data import (
    DataLoader,
    Dataset
)
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

if __name__ == "__main__":
    input_dir = "data/test_input/images"

    # set up wandb and download model
    run = wandb.init(project="owkin-chal", job_type="eval")
    artifact = run.use_artifact("model:latest")
    artifact_dir = artifact.download()
    # create output dataframe
    res = pd.DataFrame(columns=('ID', 'Target'))
    
    # set transforms
    tfms = transforms.Compose([transforms.ToTensor()])

    # set model
    model = torchvision.models.resnet34(pretrained=False) 
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(os.path.join(artifact_dir, "model.pth")))
    model.to(device="cuda")
    model.eval()

    dirs = glob(input_dir+"/*/")    
    # for each id
    for directory in dirs:
        # parse id
        ID = int(directory.split("/")[-2].replace("ID_", ""))
        print(ID)

        dataset = CustomDataSet(directory, tfms)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        positive = False
        for _, images in enumerate(loader):
            images = images.to(device="cuda")
            
            scores = model(images)
            print(scores)
            scores = scores > 0.5 
            print(scores)
            any_positives = torch.any(scores).item()
            print(any_positives)   
            break
        break
                 
        # print("files ", files)
            # run model
            # write output to dataframe
    
