import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from multiprocessing import Pool

import xml.etree.ElementTree as ET

from pathlib import Path

import torch

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image

from torchvision.datasets.folder import default_loader

import random

from tqdm import tqdm_notebook as tqdm    
class Self_Attn(nn.Module):

    """ Self attention Layer"""

    def __init__(self,in_dim,activation = 'relu'):

        super(Self_Attn,self).__init__()

        self.chanel_in = in_dim

        self.activation = activation

        

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)

        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)

        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))



        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):

        """

            inputs :

                x : input feature maps( B X C X W X H)

            returns :

                out : self attention value + input feature 

                attention: B X N X N (N is Width*Height)

        """

        m_batchsize,C,width ,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)

        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)

        energy =  torch.bmm(proj_query,proj_key) # transpose check

        attention = self.softmax(energy) # BX (N) X (N) 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N



        out = torch.bmm(proj_value,attention.permute(0,2,1) )

        out = out.view(m_batchsize,C,width,height)

        

        out = self.gamma*out + x

        return out  #,attention

    

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )



    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)



class SeparableConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):

        super(SeparableConv2d,self).__init__()



        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias))

        self.pointwise = nn.utils.spectral_norm(nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias))

    

    def forward(self,x):

        x = self.conv1(x)

        x = self.pointwise(x)

        return x

    

class pixelwise_norm_layer(nn.Module):

    def __init__(self):

        super(pixelwise_norm_layer, self).__init__()

        self.eps = 1e-8



    def forward(self, x):

        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5



class ResidualBlock(nn.Module):

    def __init__(self, in_features):

        super(ResidualBlock, self).__init__()



        conv_block = [

            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),

            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),

            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),

        ]



        self.conv_block = nn.Sequential(*conv_block)



    def forward(self, x):

        return x + self.conv_block(x)
class Generator(nn.Module):

    def __init__(self, nz=128, channels=3):

        super(Generator, self).__init__()

        

        self.nz = nz

        self.channels = channels

        

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):

            block = [

                nn.utils.spectral_norm(nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)),

                pixelwise_norm_layer(),

                nn.LeakyReLU(0.01),

            ]

            return block



        self.model = nn.Sequential(

            *convlayer(self.nz, 1024, 4, 1, 0), # Fully connected layer via convolution.

            Self_Attn(1024),

            *convlayer(1024, 512, 4, 2, 1),

            *convlayer(512, 256, 4, 2, 1),

            *convlayer(256, 128, 4, 2, 1),

            Self_Attn(128),

            *convlayer(128, 64, 4, 2, 1),

            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),

            nn.Tanh()

        )

    def forward(self, z):

        z = z.view(-1, self.nz, 1, 1)

        img = self.model(z)

        return img



class Discriminator(nn.Module):

    def __init__(self, channels=3):

        super(Discriminator, self).__init__()

        

        self.channels = channels



        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):

            block = [nn.utils.spectral_norm(nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False))]

            if bn:

                block.append(nn.BatchNorm2d(n_output))

            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        

        self.head1 = nn.Sequential(

            *convlayer(self.channels, 32, 4, 2, 1),

            nn.LeakyReLU(0.2, inplace=False),

        )

        self.head2 = nn.Sequential(

            *convlayer(32, 64, 4, 2, 1),

            nn.LeakyReLU(0.2, inplace=False),

            *convlayer(64, 128, 4, 2, 1, bn=True),

        )

        

        self.model = nn.Sequential(

            *convlayer(128, 256, 4, 2, 1, bn=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.

        )



    def forward(self, imgs):

        imgs = self.head1(imgs)

        imgs = self.head2(imgs)

        logits = self.model(imgs)

        # out = torch.sigmoid(logits)

        return logits.view(-1, 1)

    

    def feature(self, imgs):

        feature_1 = self.head1(imgs)

        feature_2 = self.head2(feature_1)

        return feature_1, feature_2
batch_size = 32

g_lr = 0.0005

d_lr = 0.0005

beta1 = 0.5

epochs = 350



real_label = 0.95

fake_label = 0

nz = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DogsDataset(torch.utils.data.Dataset):

    

    def __init__(self, root, annotation_root, transform=None,

                 target_transform=None, loader=default_loader, n_process=4):

        self.root = Path(root)

        self.annotation_root = Path(annotation_root)

        self.transform = transform

        self.target_transform = target_transform

        self.loader = loader

        self.imgs = self.cut_out_dogs(n_process)



    def _get_annotation_path(self, img_path):

        dog = Path(img_path).stem

        breed = dog.split('_')[0]

        breed_dir = next(self.annotation_root.glob(f'{breed}-*'))

        return breed_dir / dog

    

    @staticmethod

    def _get_dog_box(annotation_path):

        tree = ET.parse(annotation_path)

        root = tree.getroot()

        objects = root.findall('object')

        for o in objects:

            bndbox = o.find('bndbox')

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

            yield (xmin, ymin, xmax, ymax)

            

    def crop_dog(self, path):

        imgs = []

        annotation_path = self._get_annotation_path(path)

        for bndbox in self._get_dog_box(annotation_path):

            img = self.loader(path)

            img_ = img.crop(bndbox)

            if np.sum(img_) != 0:

                img = img_

            imgs.append(img)

        return imgs

    

    def cut_out_dogs(self, n_process):

        with Pool(n_process) as p:

            imgs = p.map(self.crop_dog, self.root.iterdir())

        return imgs

    

    def __getitem__(self, index):

        samples = random.choice(self.imgs[index])

        if self.transform is not None:

            samples = self.transform(samples)

        return samples

    

    def __len__(self):

        return len(self.imgs)
random_transforms = [

    #transforms.ColorJitter(brightness=0.75, contrast=0.75, saturation=0.75, hue=0.51), 

    transforms.RandomRotation(degrees=5)]

transform = transforms.Compose([transforms.Resize(64),

                                transforms.CenterCrop(64),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomApply(random_transforms, p=0.3),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

TRAIN_DIR = Path('../input/all-dogs/')

ANNOTATION_DIR = Path('../input/annotation/Annotation/')

#train_data = datasets.ImageFolder('../input/all-dogs/', transform=transform)

train_data = DogsDataset(TRAIN_DIR / 'all-dogs/', ANNOTATION_DIR, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,

                                           batch_size=batch_size, num_workers=4)

                                           

imgs = next(iter(train_loader))

imgs = imgs.numpy().transpose(0, 2, 3, 1)

plt.imshow(imgs[0])

plt.show()
netG = Generator(nz).to(device)

netD = Discriminator().to(device)



criterion = nn.BCEWithLogitsLoss()



optimizerD = optim.Adam(netD.parameters(), lr=g_lr, betas=(beta1, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=d_lr, betas=(beta1, 0.999))



#fixed_noise = torch.randn(25, nz, 1, 1, device=device)

def show_generated_img(netG):

    noise = torch.randn(1, nz, 1, 1, device=device)

    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)

    gen_image = (gen_image.numpy().transpose(1, 2, 0) + 1)/2.

    plt.imshow(gen_image)

    plt.show()
for epoch in tqdm(range(epochs)):

    avgErrD = 0.0

    avgErrG = 0.0

    avgD_x, avgD_G_z1, avgD_G_z2 = 0.0, 0.0, 0.0

    count = 0

    for ii, real_images in enumerate(train_loader):

        ############################

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        ###########################

        # train with real

        netD.zero_grad()

        real_images = real_images.to(device)

        batch_size = real_images.size(0)

        labels = torch.full((batch_size, 1), real_label, device=device)



        outputR = netD(real_images)

        errD_real = criterion(outputR, labels)

        errD_real.backward()

        D_x = outputR.mean().item()



        # train with fake

        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        fake = netG(noise)

        labels.fill_(fake_label)

        outputF = netD(fake.detach())

        errD_fake = criterion(outputF, labels)

        errD_fake.backward()

        D_G_z1 = outputF.mean().item()

        errD = errD_real + errD_fake

        #errD = (torch.mean((outputR - torch.mean(outputF) - labels) ** 2) + 

                 #torch.mean((outputF - torch.mean(outputR) + labels) ** 2)) / 2

        #errD.backward(retain_graph=True)

        optimizerD.step()

        ############################

        # (2) Update G network: maximize log(D(G(z)))

        ###########################

        netG.zero_grad()

        labels.fill_(1.0)  # fake labels are real for generator cost

        outputF = netD(fake)

        featureR1,featureR2 = netD.feature(real_images)

        featureF1, featureF2 = netD.feature(fake)

        errG = torch.mean((outputF - torch.mean(outputR.detach()) - labels) ** 2)

        errG += 0.01* torch.mean((featureF1 - torch.mean(featureR1.detach(),0))**2)

        errG += 0.01* torch.mean((featureF2 - torch.mean(featureR2.detach(),0))**2)

        errG.backward()

        D_G_z2 = outputF.mean().item()

        optimizerG.step()

        

        avgErrD += errD.item()

        avgErrG += errG.item()

        avgD_x += D_x

        avgD_G_z1 += D_G_z1 

        avgD_G_z2 += D_G_z2

        count += 1

    print('[%d/%d]Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'% (epoch + 1, epochs, avgErrD/count, avgErrG/count, avgD_x/count, avgD_G_z1/count, avgD_G_z2/count))

    if((epoch+1)%5 == 0):

        #noise = torch.randn(32, nz, 1, 1, device=device)

        #show_images(netG(noise),'dogs')

        show_generated_img(netG)
import zipfile



z = zipfile.PyZipFile('images.zip', mode='w')

im_batch_size = 50

n_images=10000

for i_batch in range(0, n_images, im_batch_size):

    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)

    gen_images = netG(gen_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        f = 'image_{}_{}.png'.format(i_batch, i_image)

        save_image((gen_images[i_image, :, :, :]+1)/2., f);z.write(f); os.remove(f)

z.close()