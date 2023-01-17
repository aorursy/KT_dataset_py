import argparse
import os           # OS will be used traverse through files.
import numpy as np  # Numpy will be used to do maths stuff.
import math         # Math for maths.

import torchvision.transforms as transforms  # Transforms will be used for Data Augmentation.
from torchvision.utils import save_image     # save_image will be used to save output images.

from torch.utils.data import DataLoader      # DataLoader is our Compiler which compiles
                                             # everything that revolves around preprocessing.
from torchvision import datasets             # Using this to import the MNIST Dataset. 
from torch.autograd import Variable          

# Everything below is used to define our Generator and Discriminator
import torch.nn as nn                  
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt              # Matplotlib will be used to plot output.
# Creating a new folder where our output will exist.
os.makedirs('output', exist_ok = True)
# Creating a new folder where our we have to put our input dataset.
os.makedirs('input', exist_ok = True)
# image_shape refers to the shape of the input images.
img_shape = (1, 28, 28)
# Generator refers to the part of our Network which Generates noise.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()     
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.in1 = nn.BatchNorm1d(128)
        self.in2 = nn.BatchNorm1d(512)
        self.in3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.in2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.in3(self.fc3(x)), 0.2)
        x = F.leaky_relu(self.fc4(x))
        return x.view(x.shape[0],*img_shape)
    
# Discriminator will attack the noise produced by the Generator for not being like the input images.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu((self.fc1(x)), 0.2)
        x = F.leaky_relu((self.fc2(x)), 0.2)
        x = F.leaky_relu((self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x
# Defining the Loss Function:
loss_func = torch.nn.BCELoss()
# Making the Generator and Discriminator classes usable:
generator = Generator()

discriminator = Discriminator()
data = torch.utils.data.DataLoader(
    datasets.MNIST('./input/',download = True,           # Downloading the Dataset.
                   train=True,                           # Yes, we will be training this bad boy.
              transform = transforms.Compose([           # Finally, transforming the data.
                  transforms.ToTensor(),                 # Converting the images into Tensors.
                  transforms.Normalize([0.5],[0.5])])),  # Normalizing the Tensors.
    batch_size = 64, shuffle = True)                     # Defining the batchsize and shuffling the data. 
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_func.cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002,betas=(0.4,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,betas=(0.4,0.999))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for epoch in range(150):
    for i, (imgs, _) in enumerate(data):

        #ground truths
        val = Tensor(imgs.size(0), 1).fill_(1.0)
        fake = Tensor(imgs.size(0), 1).fill_(0.0)

        real_imgs = imgs.cuda()


        optimizer_G.zero_grad()

        gen_input = Tensor(np.random.normal(0, 1, (imgs.shape[0],100)))

        gen = generator(gen_input)

        #measure of generator's ability to fool discriminator
        g_loss = loss_func(discriminator(gen), val)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = loss_func(discriminator(real_imgs), val)
        fake_loss = loss_func(discriminator(gen.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 150, i, len(data),
                                                            d_loss.item(), g_loss.item()))
        
        total_batch = epoch * len(data) + i
        if total_batch % 400 == 0:
            save_image(gen.data[:25], 'output/%d.png' % total_batch, nrow=5, normalize=True)

# This literally nothing but noise becuase it is the first output.
plt.imshow(plt.imread("./output/0.png"))
# These are improving I guess:
plt.imshow(plt.imread("./output/10000.png"))
# Lets see it again:
plt.imshow(plt.imread("./output/20000.png"))
plt.imshow(plt.imread("./output/60000.png"))
plt.imshow(plt.imread("./output/70000.png"))
plt.imshow(plt.imread("./output/80000.png"))
plt.imshow(plt.imread("./output/100000.png"))
plt.imshow(plt.imread("./output/110000.png"))