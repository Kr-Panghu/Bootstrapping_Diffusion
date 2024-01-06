#use this code to evaluate the img from a model
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
from PIL import Image
import argparse
from gan import Discriminator
device = torch.device("cuda:0")
#need to be same with ndf in gan.py
ndf = 64
netD =  Discriminator(1).to(device)
netD.load_state_dict(torch.load(f'netD_{ndf}_n1.pth'))
dataroot = "/home/cs3964_group2/yuxiaoyang/final_proj/fake"
 
eval_model='Diffusion'
# Number of workers for dataloader
workers = 2
 
batch_size = 100
 
image_size = 64

nz = 100
ngpu = 1
real_label = 1.
fake_label = 0.
parser = argparse.ArgumentParser()
parser.add_argument("--model",default='face')# specify which model to evaluate
args = parser.parse_args()
dataroot=os.path.join(dataroot,args.model)
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
for i, data in enumerate(dataloader, 0):
    imgs = data[0].to(device)
    b_size = imgs.size(0)
    label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
    output = netD(imgs).view(-1)
    print('acc:',output.mean())
    print('std:',output.std())
  
    # Forward pass real batch through D
  