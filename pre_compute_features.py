import os
import torch
import wandb
import torchvision
from torch import nn
from histo_dataset import HistoDataset

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    o_dir = "r18_features"
    
    os.makedirs(o_dir)    
    
    run = wandb.init(project="owkin-chal", job_type="pre_compute")
    artifact = run.use_artifact("model:v7")
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "model.pth")

    model = torchvision.models.resnet18(pretrained=False) 
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(model_path)) 
    model.fc = Identity()

    tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset = HistoDataset("data", transform=tfms) 

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    ) 

    model.cuda()

    batch_nb = len(loader)

    with torch.no_grad():
        for batch_idx, (ID, data, targets) in enumerate(loader): 
            print(batch_idx, "/", batch_nb)
            data = data.to(device="cuda")
            targets = targets.to(device="cuda").to(torch.float32)

            features = None
            
            for i in range(data.shape[1]):
                current_image = data[:,i,:,:]
                curr_scores = model(current_image)
                
                if features == None:
                    features = curr_scores
                else:
                    features = torch.cat((features, curr_scores), 0)

            # print("features size ", features.size())
            # print("targets ", targets)

            features = features.cpu().numpy()
            features.dump(os.path.join(o_dir, "ID_{idx}.npy".format(idx=str(int(ID)).zfill(3))))
