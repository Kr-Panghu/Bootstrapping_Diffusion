# run this code to generate fake img from a model
from gan import Generator
import gan
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision.utils import save_image
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
ndf = 64
netG =  Generator(1).to(device)
netG.load_state_dict(torch.load(f'netG_{ndf}_cmp.pth'))#the model name for generating the img
model_name='Gan'

workers = 2
 
batch_size = 64
 
image_size = 64

nz = 100
 
ngpu = 1
dataroot = "/home/cs3964_group2/data"
save_path=f"./fake/{model_name}_{ndf}_n1/data" #where to save imgs
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
netG.eval()
b_size = 1
fixed_noise = torch.randn(100, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
 
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
if not os.path.exists(save_path):
        os.makedirs(save_path)
for i,img in enumerate(fake):
    save_image(img,f'{save_path}/{i + 1}.png',normalize=True)
   # img.save(f'{save_path}/output_image_{i + 1}.png')

 
 
 