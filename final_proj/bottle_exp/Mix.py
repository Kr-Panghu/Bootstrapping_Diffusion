# run this code to generate fake img from a model
from gan import Generator
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

# original saved file with DataParallel
state_dict = torch.load(f'netG.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v


device = torch.device("cuda:0")
netG =  Generator(1).to(device)
netG.load_state_dict(torch.load(new_state_dict))
model_name='Gan'
# Number of workers for dataloader
workers = 2
 
batch_size = 64
 
image_size = 64

nz = 100
 
ngpu = 1
# dataroot = "/home/cs3964_group2/data"
save_path=f"/home/szchen/water_glass2/water_glass"
# dataset = dset.ImageFolder(root=dataroot,
#                             transform=transforms.Compose([
#                                 transforms.Resize(image_size),
#                                 transforms.CenterCrop(image_size),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ]))
#     # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                             shuffle=True, num_workers=workers)
netG.eval()
b_size = 1
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
if not os.path.exists(save_path):
        os.makedirs(save_path)
for i,img in enumerate(fake):
 
    save_image(img,f'{save_path}/{i + 101}.png',normalize=True)
   # img.save(f'{save_path}/output_image_{i + 1}.png')

res =vutils.make_grid(fake, padding=2, normalize=True)
 
# plt.subplot(1,2,1)
# plt.axis("off")
# plt.title("Real Images")
# real_batch = next(iter(dataloader))
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(res,(1,2,0)))

plt.savefig('Mixed_bottles.png')