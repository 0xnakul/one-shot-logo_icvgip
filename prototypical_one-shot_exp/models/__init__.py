from torch import ModuleDict
from models.vaeIdsiaStn import *
from torchvision.models import resnet50, resnet18
import torch.nn as nn

def get_model(name, n_classes=None, txt_dim=0):
    if 'vaeIdsia' in name:
        model = _get_model_instance(name)

    if name is 'vaeIdsiaStn':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150]) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (stn1 + stn3) with random initialization!')

    if name is 'vaeIdsia':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=None, param2=None, param3 = None) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (without stns) with random initialization!')
    
    if name == 'resnet50p':
        model = resnet50(True)
        model.fc = nn.Identity()
        fc = nn.Sequential(
            nn.Linear(2048+txt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

        return nn.ModuleList([model, fc])
    
    if name == 'resnet50r':
        model = resnet50()
        
        model.fc = nn.Sequential(
            nn.Linear(2048+txt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
    
    if name == 'resnet18p':
        model = resnet18(True)
        model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    return model

def _get_model_instance(name):
    try:
        return {
            'vaeIdsiaStn' : VAEIdsia,
            'vaeIdsia' : VAEIdsia,
        }[name]
    except:
        print('Model {} not available'.format(name))
