import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# torch

import torch

from torch import nn, optim

import torch.nn.functional as F

import torchvision.utils as utils

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from torchvision.utils import save_image

from torch.autograd import Variable

from torchvision.models.vgg import vgg16

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize



import argparse

import time

from math import log10

from PIL import Image

from tqdm import tqdm_notebook as tqdm

import xml.etree.ElementTree as ET # for parsing XML

import math

from math import exp



import matplotlib

import matplotlib.pyplot as plt # plotting



import shutil # used for making zip archives of output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SRGenerator(nn.Module):

    def __init__(self, scale_factor):

        upsample_block_num = int(math.log(scale_factor, 2))



        super(SRGenerator, self).__init__()

        self.block1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=9, padding=4),

            nn.PReLU()

        )

        self.block2 = ResidualBlock(64)

        self.block3 = ResidualBlock(64)

        self.block4 = ResidualBlock(64)

        self.block5 = ResidualBlock(64)

        self.block6 = ResidualBlock(64)

        self.block7 = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64)

        )

        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]

        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))

        self.block8 = nn.Sequential(*block8)



    def forward(self, x):

        block1 = self.block1(x)

        block2 = self.block2(block1)

        block3 = self.block3(block2)

        block4 = self.block4(block3)

        block5 = self.block5(block4)

        block6 = self.block6(block5)

        block7 = self.block7(block6)

        block8 = self.block8(block1 + block7)



        return (torch.tanh(block8) + 1) / 2





class SRDiscriminator(nn.Module):

    def __init__(self):

        super(SRDiscriminator, self).__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),

            nn.LeakyReLU(0.2),



            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2),



            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),



            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),



            nn.Conv2d(128, 256, kernel_size=3, padding=1),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2),



            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2),



            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2),



            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2),



            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(512, 1024, kernel_size=1),

            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, kernel_size=1)

        )



    def forward(self, x):

        batch_size = x.size(0)

        return torch.sigmoid(self.net(x).view(batch_size))





class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)

        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(channels)



    def forward(self, x):

        residual = self.conv1(x)

        residual = self.bn1(residual)

        residual = self.prelu(residual)

        residual = self.conv2(residual)

        residual = self.bn2(residual)



        return x + residual





class UpsampleBLock(nn.Module):

    def __init__(self, in_channels, up_scale):

        super(UpsampleBLock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)

        self.prelu = nn.PReLU()



    def forward(self, x):

        x = self.conv(x)

        x = self.pixel_shuffle(x)

        x = self.prelu(x)

        return x
class GeneratorLoss(nn.Module):

    def __init__(self):

        super(GeneratorLoss, self).__init__()

        vgg = vgg16(pretrained=True)

        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()

        for param in loss_network.parameters():

            param.requires_grad = False

        self.loss_network = loss_network

        self.mse_loss = nn.MSELoss()

        self.tv_loss = TVLoss()



    def forward(self, out_labels, out_images, target_images):

        # Adversarial Loss

        adversarial_loss = torch.mean(1 - out_labels)

        # Perception Loss

        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Image Loss

        image_loss = self.mse_loss(out_images, target_images)

        # TV Loss

        tv_loss = self.tv_loss(out_images)

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss





class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):

        super(TVLoss, self).__init__()

        self.tv_loss_weight = tv_loss_weight



    def forward(self, x):

        batch_size = x.size()[0]

        h_x = x.size()[2]

        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])

        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size



    @staticmethod

    def tensor_size(t):

        return t.size()[1] * t.size()[2] * t.size()[3]





if __name__ == "__main__":

    g_loss = GeneratorLoss()

    print(g_loss)
def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])





def calculate_valid_crop_size(crop_size, upscale_factor):

    return crop_size - (crop_size % upscale_factor)





def train_hr_transform(crop_size):

    return Compose([

        RandomCrop(crop_size),

        ToTensor(),

    ])





def train_lr_transform(crop_size, upscale_factor):

    return Compose([

        ToPILImage(),

        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),

        ToTensor()

    ])





def display_transform():

    return Compose([

        ToPILImage(),

        Resize(400),

        CenterCrop(400),

        ToTensor()

    ])





class TrainDatasetFromFolder(Dataset):

    def __init__(self, dataset_dir, crop_size, upscale_factor):

        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        self.hr_transform = train_hr_transform(crop_size)

        self.lr_transform = train_lr_transform(crop_size, upscale_factor)



    def __getitem__(self, index):

        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))

        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image



    def __len__(self):

        return len(self.image_filenames)





class ValDatasetFromFolder(Dataset):

    def __init__(self, dataset_dir, upscale_factor):

        super(ValDatasetFromFolder, self).__init__()

        self.upscale_factor = upscale_factor

        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]



    def __getitem__(self, index):

        hr_image = Image.open(self.image_filenames[index])

        w, h = hr_image.size

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)

        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        hr_image = CenterCrop(crop_size)(hr_image)

        lr_image = lr_scale(hr_image)

        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)



    def __len__(self):

        return len(self.image_filenames)





class TestDatasetFromFolder(Dataset):

    def __init__(self, dataset_dir, upscale_factor):

        super(TestDatasetFromFolder, self).__init__()

        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'

        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'

        self.upscale_factor = upscale_factor

        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_image_file(x)]

        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_image_file(x)]



    def __getitem__(self, index):

        image_name = self.lr_filenames[index].split('/')[-1]

        lr_image = Image.open(self.lr_filenames[index])

        w, h = lr_image.size

        hr_image = Image.open(self.hr_filenames[index])

        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)

        hr_restore_img = hr_scale(lr_image)

        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)



    def __len__(self):

        return len(self.lr_filenames)
def ssim(img1, img2, window_size = 11, size_average = True):

    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)

    

    if img1.is_cuda:

        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    

    return _ssim(img1, img2, window, window_size, channel, size_average)



def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])

    return gauss/gauss.sum()



def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window



def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)

    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)



    mu1_sq = mu1.pow(2)

    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2



    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq

    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq

    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2



    C1 = 0.01**2

    C2 = 0.03**2



    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))



    if size_average:

        return ssim_map.mean()

    else:

        return ssim_map.mean(1).mean(1).mean(1)
#for gen_img_nr in range(100, 800):

    UPSCALE_FACTOR = 4

    TEST_MODE = True

    IMAGE_NAME = '../input/generated-image/1_gen_img.png'

    MODEL_NAME = '../input/srgan-trained/netG_epoch_4_100.pth'



    model = SRGenerator(UPSCALE_FACTOR).eval()

    if TEST_MODE:

        model.cuda()

        model.load_state_dict(torch.load(MODEL_NAME))

    else:

        model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))



    image = Image.open(IMAGE_NAME).convert('RGB')

    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

    if TEST_MODE:

        image = image.cuda()



    start = time.clock()

    out = model(image)

    elapsed = (time.clock() - start)

    print('cost' + str(elapsed) + 's')

    out_img = ToPILImage()(out[0].data.cpu())

    plt.imshow(out_img)

    out_img.save("generatedimage.png")    