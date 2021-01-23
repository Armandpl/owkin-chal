import torch
import torchvision
from torch import nn
from torchvision import models

class Resnet18Rnn(nn.Module):

    def __init__(self, num_classes=1, dr_rate=0.1, rnn_hidden_size=100, rnn_num_layers=1, model_path="_ckpt_epoch_9.ckpt"):
        super(Resnet18Rnn, self).__init__()
        num_classes = num_classes
        dr_rate= dr_rate
        rnn_hidden_size = rnn_hidden_size
        rnn_num_layers = rnn_num_layers
        
        baseModel = models.resnet18(pretrained=False)
        # state = torch.load(model_path, map_location='cuda:0')

        # state_dict = state['state_dict']
        # for key in list(state_dict.keys()):
        # j    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        # baseModel = self.load_model_weights(baseModel, state_dict)

        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel

        self.freeze_base_model()

        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
    def load_state_dict_for_base(self, path):
        self.baseModel.fc = nn.Linear(512, 1)
        self.baseModel.load_state_dict(torch.load(path))
        self.baseModel.fc = Identity()
    
    def freeze_base_model(self):
        for param in self.baseModel.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        for param in self.baseModel.parameters():
            param.requires_grad = True

    def load_model_weights(self, model, weights):

        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
