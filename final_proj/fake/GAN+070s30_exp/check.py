# Modified from 'szchen/check.py'
from gan import Generator
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
device = torch.device("cuda:0")
nz = 100
netG =  Generator(1).to(device)
netG.load_state_dict(torch.load('netG_32_n1.pth'))

netG.eval()
b_size = 1
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
res =vutils.make_grid(fake, padding=2, normalize=True)
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(res,(1,2,0)))
plt.savefig('Ganres_bottle.png')