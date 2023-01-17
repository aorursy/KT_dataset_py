# can comment out after executing

# !unzip processed_celeba_small.zip
#import os

#for dirname in os.walk('/kaggle/input'):

   # print(dirname[:15])
data_dir = '/kaggle/input/facegenerationdata/processed_celeba_small/processed_celeba_small/'



# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

# copyfile(src = "/kaggle/input/assets/helper.py", dst = "../working/helper.py")

copyfile(src = "/kaggle/input/facegenerationdata/assets/problem_unittests.py", dst = "../working/problem_unittests.py")

"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import pickle as pkl

import matplotlib.pyplot as plt

import numpy as np

import problem_unittests as tests

#import helper



%matplotlib inline
# necessary imports



import torch

from torchvision import datasets

from torchvision import transforms
def get_dataloader(batch_size, image_size, data_dir='/kaggle/input/facegenerationdata/processed_celeba_small/processed_celeba_small/'):

    """

    Batch the neural network data using DataLoader

    :param batch_size: The size of each batch; the number of images in a batch

    :param img_size: The square size of the image data (x, y)

    :param data_dir: Directory where image data is located

    :return: DataLoader with batched data

    """

    

    # This Below Code is written for Center Cropping the Images               

    transform = transforms.Compose([

        transforms.Resize(image_size),

        transforms.CenterCrop(image_size),

        transforms.ToTensor(),

        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    ])               

    

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    

    # TODO: Implement Function and Return a DataLoader

    image_transforms = transforms.Compose([transforms.Resize(image_size),

                                           transforms.ToTensor(),

                                          ])

    dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir, transform=image_transforms)

                                             , shuffle=True, batch_size=batch_size)

    

    return dataloader
# Define function hyperparameters

batch_size = 30

img_size = 32



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# Call your function and get a dataloader

celeba_train_loader = get_dataloader(batch_size, img_size)

celeba_train_loader
# helper display function

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# obtain one batch of training images

dataiter = iter(celeba_train_loader)

images, _ = dataiter.next() # _ for no labels



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(20, 4))

plot_size=20

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])
# TODO: Complete the scale function

def scale(x, feature_range=(-1, 1)):

    

    ''' Scale takes in an image x and returns that image, scaled

       with a feature_range of pixel values from -1 to 1. 

       This function assumes that the input x is already scaled from 0-1.'''

    

    # assume x is scaled to (0, 1)

    # scale to feature_range and return scaled x

    min, max = feature_range

    

   # Print(x)

    x = x * (max - min) + min

    # You can Write the Above Syntax as here ---> x*(feature_range[1] - feature_range[0]) + feature_range[0]

    

    return x

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# check scaled range

# should be close to -1 to 1

img = images[0]

scaled_img = scale(img)



print('Min: ', scaled_img.min())

print('Max: ', scaled_img.max())
import torch.nn as nn

import torch.nn.functional as F
# Hepler Convoluitonal Function

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    

    """

        Creates a Convolutional Layer, with Optional Batch Normalization.

    """

    

    layers = []

    conv_layer = nn.Conv2d(in_channels, out_channels, 

                           kernel_size, stride, padding, bias=False)

    

    # Append Convolutional Layer

    layers.append(conv_layer)



    if batch_norm:

        

        # Append Batch Normalization Layer

        layers.append(nn.BatchNorm2d(out_channels))

    

    # Using Sequential Container

    return nn.Sequential(*layers)
class Discriminator(nn.Module):



    def __init__(self, conv_dim):

        """

        Initialize the Discriminator Module

        :param conv_dim: The depth of the first convolutional layer

        """

        super(Discriminator, self).__init__()



        # complete init function

        self.conv_dim = conv_dim

        

        # Define All Convolutional Layers

        # Should Accept RGB as Input & Output a Single Value

        

        # For Input of 32x32 & For First Layer, no batch_norm

        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)

        

        # For Output of 16x16

        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        

        # For Output of 8x8

        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        

        # For Output of 4x4

        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)

        

        # For Output of 2x2

        # Now Final, Fully-Connected Layer

        self.fc = nn.Linear(conv_dim*8*2*2, 1)

        

        # For Final Output Apply Sigmoid Activation Function

        self.out = nn.Sigmoid()

        

        # Apply Dropout If Needed

        self.dropout = nn.Dropout(0.5)



        

    def forward(self, x):

        """

        Forward propagation of the neural network

        :param x: The input to the neural network     

        :return: Discriminator logits; the output of the neural network

        """

       # Define Feed-Forward Behaviour

        # All Hidden Layers & ReLu Activation Function

        out = F.leaky_relu(self.conv1(x), 0.2)

        out = F.leaky_relu(self.conv2(out), 0.2)

        out = F.leaky_relu(self.conv3(out), 0.2)

        out = F.leaky_relu(self.conv4(out), 0.2)

        

        # Flatten the Ouput

        out = out.view(-1, self.conv_dim*8*2*2)

        

        # Final Output Layer

        x = self.fc(out)

        

        # Return the Final Ouput

        return x





"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_discriminator(Discriminator)
# For Deconvolutional Function



def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    

    layers=[]

    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,

                                             kernel_size, stride, padding, bias=False)

    

    # Append Transpose Convolutional Layer

    layers.append(transpose_conv_layer)

    

    if batch_norm:

        

        # Append Batch Normalization Layer

        layers.append(nn.BatchNorm2d(out_channels))

    

    # Using Sequential Container

    return nn.Sequential(*layers)
class Generator(nn.Module):

    

    def __init__(self, z_size, conv_dim):

        """

        Initialize the Generator Module

        :param z_size: The length of the input latent vector, z

        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer

        """

        super(Generator, self).__init__()



        # complete init function

        self.conv_dim = conv_dim

        

        # First, Fully-Connected Layer

        self.fc = nn.Linear(z_size, conv_dim*8*2*2)



        # Transpose Covlolutional Layer

        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)

        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)

        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)

        self.deconv4 = deconv(conv_dim, 3, 4, batch_norm=False)

        # print('z_size', z_size) Find what does this Means



    def forward(self, x):

        """

        Forward Propogation of the Neural Network

        :param x: The input to the Neural Network     

        :return: A 32x32x3 Tensor Image as Output

        """

        # Define Feed-Forward Behaviour

        # Fully-Connected Layer & Reshape the deconv Layer

        out = self.fc(x)

        

        # View the Ouput of the Fully-Connected & Reshape (batch_size, depth, 4, 4)

        out = out.view(-1, self.conv_dim*8, 2, 2)

        

        # Hidden Transpose Convolutional Layers

        # Apply the ReLu Activation Function Output

        out = F.relu(self.deconv1(out))

        out = F.relu(self.deconv2(out))

        out = F.relu(self.deconv3(out))

        

        # Last Layer of the deconv

        # Apply 'tanh' Activation Function to the Output

        out = self.deconv4(out)

        x = torch.tanh(out)

        

        # Return the Final Output

        return x



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_generator(Generator)
def weights_init_normal(m):

    """

    Applies initial weights to certain layers in a model .

    The weights are taken from a normal distribution 

    with mean = 0, std dev = 0.02.

    :param m: A module or layer in a network    

    """

    

    # Classname will be something like: `Conv`, `BatchNorm2d`, `Linear`, etc.

    classname = m.__class__.__name__

    

    # TODO: Apply Initial Weights to Convolutional & Linear Layers

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        m.weight.data.normal_(0.0, 0.02)

        

        # The Bias Terms, if they exist, Set to 0

        if hasattr(m, 'bias') and m.bias is not None:

            m.bias.data.zero_()

    
"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

def build_network(d_conv_dim, g_conv_dim, z_size):

    

    # define discriminator and generator

    D = Discriminator(d_conv_dim)

    G = Generator(z_size=z_size, conv_dim=g_conv_dim)



    # initialize model weights

    D.apply(weights_init_normal)

    G.apply(weights_init_normal)



    print(D)

    print()

    print(G)

    

    return D, G
# Define model hyperparams

d_conv_dim = 64

g_conv_dim = 64

z_size = 100



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

D, G = build_network(d_conv_dim, g_conv_dim, z_size)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import torch



# Check for a GPU if is Available

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('No GPU found. Please use a GPU to train your neural network.')

else:

    print('Training on GPU!')
import random

from random import randrange, uniform



def real_loss(D_out, smooth=False):

    

    '''

    Calculates How Close Discriminator Outputs are to Being Real.

    param, D_out: Discriminator logits

    return: real loss

    '''

    batch_size = D_out.size(0)

    

    # Label Smoothing

    if smooth:

        # Smooth, Real Labels = 0.9

        labels = torch.ones(batch_size)*0.9

    else:

        # Real Labels = 1

        labels = torch.ones(batch_size)

    

    # Move Labels to GPU if is Available

    if train_on_gpu:

        labels = labels.cuda()

    

    # Binary Cross-Entropy with Logits Loss

    criterion = nn.BCEWithLogitsLoss()

    

    # Calculate Loss

    loss = criterion(D_out.squeeze(), labels)

    

    # Return the Calculated Loss

    return loss



def fake_loss(D_out):

    '''

    Calculates How Close Discriminator Outputs are to Being Fake.

    param, D_out: Discriminator Logits

    return: Fake Loss

    '''

    batch_size = D_out.size(0)

    

    # Fake Labels = 0

    labels = torch.zeros(batch_size)

    

    # Move Labels to GPU if is Available

    if train_on_gpu:

        labels = labels.cuda()

    

    # Binary Cross-Entropy with Logits Loss

    criterion = nn.BCEWithLogitsLoss()

    

    # Calculate Loss

    loss = criterion(D_out.squeeze(), labels)

    

    # Return the Calculated Loss

    return loss
import torch.optim as optim



# Create Optimizers for the Discriminator D and Generator G

d_optimizer = optim.Adam(D.parameters(), lr=0.0005, betas=(0.4, 0.999))

g_optimizer = optim.Adam(G.parameters(), lr=0.0005, betas=(0.4, 0.999))
def train(D, G, n_epochs, print_every=50):

    '''Trains adversarial networks for some number of epochs

       param, D: the discriminator network

       param, G: the generator network

       param, n_epochs: number of epochs to train for

       param, print_every: when to print and record the models' losses

       return: D and G losses'''

    

    # move models to GPU

    if train_on_gpu:

        D.cuda()

        G.cuda()



    # keep track of loss and generated, "fake" samples

    samples = []

    losses = []



    # Get some fixed data for sampling. These are images that are held

    # constant throughout training, and allow us to inspect the model's performance

    sample_size=16

    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))

    fixed_z = torch.from_numpy(fixed_z).float()

    # move z to GPU if available

    if train_on_gpu:

        fixed_z = fixed_z.cuda()



    # epoch training loop

    for epoch in range(n_epochs):



        # batch training loop

        for batch_i, (real_images, _) in enumerate(celeba_train_loader):



            batch_size = real_images.size(0)

            real_images = scale(real_images)



            # ===============================================

            #         YOUR CODE HERE: TRAIN THE NETWORKS

            # ===============================================

            

            # 1. Train the discriminator on real and fake images

            

            d_optimizer.zero_grad()

            

            # Check If GPU is Available

            if train_on_gpu:

                real_images = real_images.cuda()

            

            # 1. Train the Discriminator on Real and Fake Images

            

            # Compute the Discriminator Losses on Real Images

            D_real = D(real_images)

            d_real_loss = real_loss(D_real)

            

            # Generate Fake Images

            z_fake = np.random.uniform(-1, 1, size=(batch_size, z_size))

            z_fake = torch.from_numpy(z_fake).float()

            

            # Move to GPU if is Available

            if train_on_gpu:

                z_fake = z_fake.cuda()

            fake_images = G(z_fake)

            

            # Compute the Discriminator Losses on Fake Images

            D_fake = D(fake_images)

            d_fake_loss = fake_loss(D_fake)

            

            # Add Up Loss and Perform Back Propagation

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            d_optimizer.step()



            # 2. Train the Generator with an Adversarial Loss

            g_optimizer.zero_grad()

            

            # Generate Fake Images

            z_fake = np.random.uniform(-1, 1, size=(batch_size, z_size))

            z_fake = torch.from_numpy(z_fake).float()

            

            # Check If GPU is Available

            if train_on_gpu:

                z_fake = z_fake.cuda()

            fake_images = G(z_fake)

            

            # Compute the Discriminator Losses on Fake Images

            D_fake = D(fake_images)

            

            # Use Real Loss to Flip Labels

            g_loss = real_loss(D_fake, True)

            

            # Perform Back Propagation

            g_loss.backward()

            g_optimizer.step()

            

            

            # ===============================================

            #              END OF YOUR CODE

            # ===============================================



            # Print Some Loss Stats

            if batch_i % print_every == 0:

                

                # Append Discriminator Loss and Generator Loss

                losses.append((d_loss.item(), g_loss.item()))

                

                # Print Discriminator and Generator Loss

                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(

                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))





      

        ## AFTER EACH EPOCH##    

        # this code assumes your generator is named G, feel free to change the name

        # generate and save sample, fake images

        G.eval() # for generating samples

        samples_z = G(fixed_z)

        samples.append(samples_z)

        G.train() # back to training mode



    # Save training generator samples

    with open('train_samples.pkl', 'wb') as f:

        pkl.dump(samples, f)

    

    # finally return losses

    return losses
# set number of epochs 

n_epochs = 20





"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

# call training function

losses = train(D, G, n_epochs=n_epochs)
fig, ax = plt.subplots()

losses = np.array(losses)

plt.plot(losses.T[0], label='Discriminator', alpha=0.5)

plt.plot(losses.T[1], label='Generator', alpha=0.5)

plt.title("Training Losses")

plt.legend()
# helper function for viewing a list of passed in sample images

def view_samples(epoch, samples):

    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[epoch]):

        img = img.detach().cpu().numpy()

        img = np.transpose(img, (1, 2, 0))

        img = ((img + 1)*255 / (2)).astype(np.uint8)

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        im = ax.imshow(img.reshape((32,32,3)))
# Load samples from generator, taken while training

with open('train_samples.pkl', 'rb') as f:

    samples = pkl.load(f)
_ = view_samples(-1, samples)