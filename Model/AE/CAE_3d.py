import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

mse_loss = nn.MSELoss()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        print('using deep encoder')
        self.encoder = nn.Sequential(nn.Linear(6000, 2048), nn.PReLU(), nn.Linear(2048, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))
    def forward(self, x):
        x = self.encoder(x)
        return x
