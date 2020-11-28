from torch import *
from torch import nn
from torch.nn import *
from torch.nn import functional as F
import torch

def get_dig_net():
    """
    Digit CNN.
    """
    return nn.Sequential(
        nn.Conv2d(1,  32, 3), nn.ReLU(), nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, stride = 2), nn.Flatten(),
        nn.Linear(256, 100), nn.ReLU(),
        nn.Linear(100, 200), nn.ReLU(),
        nn.Linear(200, 100), nn.ReLU(),
        nn.Linear(100,  10)
    )

def get_comp_net_normal():
    """
    Simple Comparison CNN.
    """
    return nn.Sequential(
        nn.Conv2d(2, 16, 2) ,nn.ReLU(),
        nn.Conv2d(16, 32, 2) ,nn.ReLU(),
        nn.Conv2d(32, 32, 2) ,nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(800, 75), nn.ReLU(),
        nn.Linear(75, 75), nn.ReLU(),
        nn.Linear(75, 1)
    )

def get_comp_net_shared():
    """
    Weight sharing CNN.
    """
    weight_shared_layer1 = nn.Sequential(nn.Linear(75, 75), nn.ReLU())
    weight_shared_layer2 = nn.Sequential(nn.Linear(75, 75), nn.ReLU())
    return nn.Sequential(
        nn.Conv2d(2, 16, 2) ,nn.ReLU(),
        nn.Conv2d(16, 32, 2) ,nn.ReLU(),
        nn.Conv2d(32, 32, 2) ,nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(800, 75), nn.ReLU(),
        weight_shared_layer1,
        weight_shared_layer1,
        weight_shared_layer1,
        nn.Linear(75, 1)
    )

class AuxLossCnn(nn.Module):
    """
    Auxiliary Loss CNN.
    """
    def __init__(self):
        super(AuxLossCnn, self).__init__()
        self.digit_model = nn.Sequential(
            nn.Conv2d(1, 16, 4) ,nn.ReLU(),
            nn.Conv2d(16, 32, 4) ,nn.ReLU(),
            nn.Conv2d(32, 32, 4) ,nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 10)
        )
        self.comparaison_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(), 
            nn.Linear(50, 50), nn.ReLU(),  
            nn.Linear(50,  1)
        )
        
    def forward(self, x):
        x = x.view(-1, 2, 1, x.shape[2], x.shape[3])
        dig1, dig2 = self.digit_model(x[:,0]), self.digit_model(x[:,1])
        x = torch.cat([dig1, dig2], axis=1)
        x_aux = x.view(-1, 10)
        x = self.comparaison_model(x)
        return x, x_aux
    
def get_aux_loss_cnn(): return AuxLossCnn()

class SiameseNetwork(nn.Module):
    """
    Siamese CNN.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.digit_model = nn.Sequential(
            nn.Conv2d(1, 16, 4) ,nn.ReLU(),
            nn.Conv2d(16, 32, 4) ,nn.ReLU(),
            nn.Conv2d(32, 32, 4) ,nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 10)
        )
        self.comparaison_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(), 
            nn.Linear(50, 50), nn.ReLU(),  
            nn.Linear(50,  1)
        )
        
    def forward(self, x):
        x = x.view(-1, 2, 1, x.shape[2], x.shape[3])
        dig1, dig2 = self.digit_model(x[:,0]), self.digit_model(x[:,1])
        x = torch.cat([dig1, dig2], axis=1)
        x_aux = x.view(-1, 10)
        x = self.comparaison_model(x)
        return x
    
def get_siamese_net(): return SiameseNetwork()

