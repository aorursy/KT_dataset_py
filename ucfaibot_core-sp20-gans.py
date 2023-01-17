from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-gans").exists():

    DATA_DIR /= "ucfai-core-sp20-gans"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-gans/data

    DATA_DIR = Path("data")
## general imports

import numpy as np

import time

import os

import math

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML



## torch imports

import torch

import torch.nn as nn

import torch.optim as optim

import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms



from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder



import torchvision.utils as vutils

from torch.utils.data import random_split



## uncomment to use specific seed for randomly generating weights and noise

## seed = 999

## torch.manual_seed(seed)
!pip install torchsummary



try:

    import torchsummary

except:

    torchsummary = None



from tabulate import tabulate



BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"

EPOCH_TEMPLATE = "Epoch [{} / {}]:"

TEST_TEMPLATE = "Epoch [{}] Test:"



def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):

    """

    Formats an iteration. kwargs should be a variable amount of metrics=vals

    Optional Arguments:

        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])

        epochs(int): total number of epochs

        batch_i(int): current batch iteration

        num_batches(int): total number of batches

        writer(SummaryWriter): tensorboardX summary writer object

        msg(bool): if true, doesn't print but returns the message string



    if curr_epoch and epochs is defined, will format end of epoch iteration

    if batch_i and num_batches is also defined, will define a batch iteration

    if curr_epoch is only defined, defines a validation (testing) iteration

    if none of these are defined, defines a single testing iteration

    if writer is not defined, metrics are not saved to tensorboard

    """

    if curr_epoch is not None:

        if batch_i is not None and num_batches is not None and epochs is not None:

            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)

        elif epochs is not None:

            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)

        else:

            out = TEST_TEMPLATE.format(curr_epoch + 1)

    else:

        out = "Testing Results:"



    floatfmt = []

    for metric, val in kwargs.items():

        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:

            floatfmt.append(".4f")

        elif "accuracy" in metric or "acc" in metric:

            floatfmt.append(".2f")

        else:

            floatfmt.append(".6f")



        if writer and curr_epoch:

            writer.add_scalar(metric, val, curr_epoch)

        elif writer and batch_i:

            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))



    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)



    if msg:

        return out

    print(out)



def summary(model, input_dim):

    if torchsummary is None:

        raise(ModuleNotFoundError, "TorchSummary was not found!")

    torchsummary.summary(model, input_dim)
image_size = (64, 64)

batch_size = 128

num_workers = 2



## I'm sorry little one

thanos_level = 4



dataset = ImageFolder(str(DATA_DIR), transform=transforms.Compose([

                               transforms.Resize(image_size),

                               transforms.CenterCrop(image_size),

                               transforms.ToTensor(),

                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                           ]))



## comment out if you want to use whole dataset

dataset, _ = random_split(dataset, [int(len(dataset) / thanos_level), len(dataset) - int((len(dataset) / thanos_level))])



## TODO:Create the dataloader from our dataset above

# YOUR CODE HERE

raise NotImplementedError()



print("Length of dataset: {}, dataloader: {}".format(len(dataset), len(dataloader)))



## Plot some training images

real_batch = next(iter(dataloader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
def get_padding(output_dim, input_dim, kernel_size, stride):

    """

    Calculates padding given in output and input dim, and parameters of the convolutional layer



    Arguments should all be integers. Use this function to calculate padding for 1 dimesion at a time.

    Output dimensions should be the same or bigger than input dimensions



    Returns 0 if invalid arguments were passed, otherwise returns an int or tuple that represents the padding.

    """



    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2



    if padding < 0:

        return 0

    else:

        return padding



print(get_padding(32, 64, 4, 2))
def gen_block(input_channels, output_channels, kernel_size, stride, padding):

    layers = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False)]

    layers += [nn.BatchNorm2d(output_channels)]

    layers += [nn.ReLU(inplace=True)]

    

    return layers

    

class Generator(nn.Module):

    def __init__(self, channels=3, input_size=100, output_dim=64):

        super(Generator, self).__init__()

        self.channels = channels

        self.input_size = input_size

        self.output_size = output_dim

        self.layers = self.build_layers()

        

    def forward(self, x):

        return self.layers(x).squeeze()

    

    def build_layers(self):

        layers = []

        in_c = self.input_size

        out_c = self.output_size * 8

        

        # dim: out_c x 4 x 4

        layers += gen_block(in_c, out_c, 4, 1, 0)

        in_c = out_c

        out_c = self.output_size * 4

        

        # TODO: Create the next two blocks the same way the above one is created

        # Use kernel size of 4 and a stride of 2. Whats the padding?

        # YOUR CODE HERE

        raise NotImplementedError()

        # dim: out_c x 32 x 32

        layers += gen_block(in_c, out_c, 4, 2, 1)

        in_c = out_c

        out_c = self.channels

        

        # dim: out_c x 64 x 64

        # don't use batch norm in the last layer since its the output.

        layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.Tanh()]

        

        return nn.Sequential(*layers)
def discrim_block(input_channels, output_channels, kernel_size, stride, padding):

    layers = []

    layers += [nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False)]

    layers += [nn.BatchNorm2d(output_channels)]

    layers += [nn.LeakyReLU(0.2, inplace=True)]

    

    return layers



class Discriminator(nn.Module):

    def __init__(self, channels=3, input_dim=64):

        super(Discriminator, self).__init__()

        self.channels = channels

        self.input_dim = input_dim

        self.layers = self.build_layers()

        

    def forward(self, x):

        return self.layers(x).squeeze()

    

    def build_layers(self):

        layers = []

        in_c = self.channels

        out_c = self.input_dim

        

        # dim: out_c x 32 x 32

        layers += [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]

        in_c = out_c

        out_c = self.input_dim * 2

        # TODO: Create the next 2 blocks for the discriminator. Kernel size of 4 and a stride of 2

        # this is quite similar to the generator...

        # YOUR CODE HERE

        raise NotImplementedError()

        # dim: out_c x 4 x 4

        layers += discrim_block(in_c, out_c, 4, 2, 1)

        in_c = out_c

        out_c = 1

        

        # dim: 1

        layers += [nn.Conv2d(in_c, out_c, 4, 1, 0), nn.Sigmoid()]

        

        return nn.Sequential(*layers)

        
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)
gen_input = 100

gen_output = 64



gen = Generator(input_size=gen_input, output_dim=gen_output)

gen.apply(weights_init)



discrim = Discriminator(channels=3, input_dim=gen_output)

discrim.apply(weights_init)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device: {}".format(device))



gen.to(device)

discrim.to(device)



## hyperparameters from DCGAN paper

learn_rate = 0.0002



optG = optim.Adam(gen.parameters(), lr=learn_rate, betas=(0.5, 0.999))

optD = optim.Adam(discrim.parameters(), lr=learn_rate, betas=(0.5, 0.999))



## TODO: Define our criterion (loss function)

# YOUR CODE HERE

raise NotImplementedError()



fixed_noise = torch.randn(gen_output, gen_input, 1, 1, device=device)



real_label = 1

fake_label = 0



print("Generator:")

summary(gen, (gen_input, 1, 1))

print("\nDiscriminator:")

summary(discrim, (3, gen_output, gen_output))
start_time = time.time()



epochs = 5

print_step = 50



## stores our generated images at each print iteration

gen_imgs = []



for e in range(epochs):

    g_train_loss = 0

    d_train_loss = 0

    e_time = time.time()

    

    for i, data in enumerate(dataloader):



        # Train Discriminator

        

        # only need images from data, don't care about class from ImageFolder, which is data[1]

        images = data[0].to(device)

        b_size = images.size(0)

        labels = torch.full((b_size,), real_label, device=device)

        

        # train on real

        discrim.zero_grad()

        d_output = discrim(images).view(-1)

        loss_real = criterion(d_output, labels)

        loss_real.backward()

      

        # get fake data from generator

        noise = torch.randn(b_size, gen_input, 1, 1, device=device)

        fake_images = gen(noise)

        # this replaces all values in labels with fake_label, which is zero in this case

        labels.fill_(fake_label)

        

        # calculate loss and update gradients on fake

        # must detach the fake images from the computational graph of the generator, so that 

        # gradients arent updated for the generator

        d_output = discrim(fake_images.detach()).view(-1)

        loss_fake = criterion(d_output, labels)

        loss_fake.backward()

        

        # add up real and fake loss

        d_loss = loss_real + loss_fake

        

        # optimize weights after calculating real and fake loss then backprogating on each

        optD.step()

        

        d_train_loss += d_loss.item()

        

        # Train Generator

        gen.zero_grad()

        # use real labels

        labels.fill_(real_label)

        

        # get new output from discriminator for fake images, which is now updated from our above step

        d_output = discrim(fake_images).view(-1)

        # calculate the Generator's loss based on this, use real_labels since fake images should be real for generator

        # i.e the generator wants the discriminator to output real for it's fake images, so thats the target for generator

        g_loss = criterion(d_output, labels)

        

        g_loss.backward()

        optG.step()

        

        g_train_loss += g_loss.item()

        

        if i % print_step == 0:

            print_iter(curr_epoch=e, epochs=epochs, batch_i=i, num_batches=len(dataloader), d_loss=d_train_loss/(i+1), g_loss=g_train_loss/(i+1))

            # save example images

            gen.eval()

            with torch.no_grad():

                fake_images = gen(fixed_noise).detach().cpu()

                gen.train()

                gen_imgs.append(vutils.make_grid(fake_images, padding=2, normalize=True))

                

    print_iter(curr_epoch=e, epochs=epochs, d_loss=d_train_loss/(i+1), g_loss=g_train_loss/(i+1))

    print("\nEpoch {} took {:.2f} minutes.\n".format(e+1, (time.time() - e_time) / 60))

    

print("Model took {:.2f} minutes to train.".format((time.time() - start_time) / 60))
fig = plt.figure(figsize=(8,8))

plt.axis("off")

ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in gen_imgs]

ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)



HTML(ani.to_jshtml())
## Grab a batch of real images from the dataloader

real_batch = next(iter(dataloader))



## Plot the real images

plt.figure(figsize=(15,15))

plt.subplot(1,2,1)

plt.axis("off")

plt.title("Real Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))



## Plot the fake images from the last epoch

plt.subplot(1,2,2)

plt.axis("off")

plt.title("Fake Images")

plt.imshow(np.transpose(gen_imgs[-1],(1,2,0)))

plt.show()