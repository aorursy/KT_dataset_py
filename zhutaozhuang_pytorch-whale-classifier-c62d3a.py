import os

import numpy as np # linear algebra

from tqdm import tqdm # progress bar to ease my anxiety

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import time

import json

import copy



import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data import DataLoader, Dataset

from torch.autograd import Variable

import matplotlib.pyplot as plt

from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



from PIL import Image

%matplotlib inline

%load_ext autoreload

%autoreload 2
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# visualize the label map & number of classes

labels = pd.read_csv("../input/train.csv")

labels.head()

print(type(labels))
num_classes = len(labels['Id'].unique())

print(num_classes)
# define data directories 

data_dir = '../input'

train_dir = data_dir + '/train'

test_dir = data_dir + '/test'
# pytorch provides a function to convert PIL images to tensors.

# credit: https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html

pil2tensor = transforms.ToTensor()

tensor2pil = transforms.ToPILImage()



# Read the image from file. Assuming it is in the same directory.

pil_image = Image.open(train_dir + '/0a750c2e8.jpg') 

rgb_image = pil2tensor(pil_image)



# Plot the image here using matplotlib.

def plot_image(tensor):

    plt.figure()

    # imshow needs a numpy array with the channel dimension

    # as the the last dimension so we have to transpose things.

    plt.imshow(tensor.numpy().transpose(1, 2, 0))

    plt.show()



plot_image(rgb_image)



# Show the image tensor type and tensor size here.

print('Image type: ' + str(rgb_image.type()))

print('Image size: ' + str(rgb_image.size()))
# credit: https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html

from io import BytesIO

import IPython.display



r_image = rgb_image[0]

g_image = rgb_image[1]

b_image = rgb_image[2]



def show_grayscale_image(tensor):

    f = BytesIO()

    a = np.uint8(tensor.mul(255).numpy()) 

    Image.fromarray(a).save(f, 'png')

    IPython.display.display(IPython.display.Image(data = f.getvalue()))



show_grayscale_image(torch.cat((r_image, g_image, b_image), 1))
# Define transforms and data augmentation

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_data = transforms.Compose([transforms.Resize(256),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomRotation(25),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),

                                       transforms.RandomAffine(degrees=4, translate=None, scale=None, shear=None, resample=False, fillcolor=0),

                                       transforms.ToTensor(),

                                       transforms.Normalize(mean, std)])



test_data = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize(mean, std)])
# one-hot encode the labels

def encode_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    

    print(values)

    print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    print(integer_encoded)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    print(len(integer_encoded))

    print(onehot_encoded)

    print(len(onehot_encoded[0]))



    y = onehot_encoded

    return y, label_encoder



y, label_encoder = encode_labels(labels['Id'])

# this will throw a FutureWarning, ignore it.
# create a dataset out of the whale tail data

class WhaleTailDataset(Dataset):

    def __init__(self, image_folder, data_type='train', df=None, transform=None, y=None):

        self.image_folder = image_folder

        self.imgs_list = [img for img in os.listdir(image_folder)]

        self.data_type = data_type

        self.transform = transform

        self.y = y # one hot vector 

        if self.data_type == 'train':

            self.df = df.values

            print(self.df) #self.df refers to img,class array

    

    def __len__(self):

        return len(self.imgs_list)

    

    def __getitem__(self, idx):

        if self.data_type == 'train':

            img_name = os.path.join(self.image_folder, self.df[idx][0])

            label = self.y[idx] #return one hot vector 

        

        elif self.data_type == 'test':

            img_name = os.path.join(self.image_folder, self.imgs_list[idx])

            label = np.zeros((num_classes,))

        

        img = Image.open(img_name).convert('RGB')

        img = self.transform(img)

        if self.data_type == 'train':

            return img, label

        elif self.data_type == 'test':

            return img, label, self.imgs_list[idx]
# load and define the datasets

image_datasets = dict()

image_datasets['train'] = WhaleTailDataset(image_folder=train_dir, data_type='train', df=labels, transform=train_data, y=y)

image_datasets['test'] = WhaleTailDataset(image_folder=test_dir, data_type='test', transform=test_data)
# define data & batch loaders

train_size = 512

test_size = 32

num_workers = 0



dataloaders = dict()

dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=train_size, num_workers=num_workers)

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=test_size, num_workers=num_workers)
# view data statistics by type

print('Number of training images: ', len(image_datasets['train']))

print('Number of test images: ', len(image_datasets['test']))
# view tensor size 

dataiter = iter(dataloaders['train'])

images, labels = dataiter.next()



print('Batch shape: ', images.size())
# define pre-trained model

model = models.resnet152(pretrained=True)



# freeze parameters

for param in model.parameters():

    param.requires_grad = False

    

print(model)    
# define new untrained network

classifier = nn.Sequential(nn.Linear(2048, 1024),

                         nn.ReLU(),

                         nn.Dropout(0.5),

                         nn.Linear(1024, 512),

                         nn.ReLU(),

                         nn.Dropout(0.5),

                         nn.Linear(512, 5005),

                         nn.LogSoftmax(dim=1)

                        )

model.fc = classifier
# define hyperperameters

from torch.optim import lr_scheduler



num_epochs = 6

learning_rate = 0.001



criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# train the model

import matplotlib.pyplot as plt

model = model.cuda()

train_loss = []

for epoch in range(1, num_epochs+1):



    

    for batch_i, (data, target) in tqdm(enumerate(dataloaders['train']), total = len(dataloaders['train'])):

        data, target = data.cuda(), target.cuda()



        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target.float())

        train_loss.append(loss.item())



        loss.backward()

        optimizer.step()

    

    scheduler.step()

    

    print(f'Epoch - {epoch} // Training Loss: {np.mean(train_loss):.4f}')



print(train_loss)

plt.figure()

plt.plot(train_loss)

plt.show()
sub = pd.read_csv('../input/sample_submission.csv')



model.eval()

for (data, target, name) in tqdm(dataloaders['test']):

    data = data.cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    for i, (e, n) in enumerate(list(zip(output, name))):

        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(label_encoder.inverse_transform(e.argsort()[-5:][::-1]))

        

sub.to_csv('submission.csv', index=False)