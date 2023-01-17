import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt

from glob import glob

import cv2 as cv

import time



import pandas as pd

import numpy as np



import os

root = "../input/rsscn7/RSSCN7-master"

print(os.listdir(root))

classDict = {'aGrass': 0, 'bField': 1, 'cIndustry': 2, 'dRiverLake': 3, 'eForest': 4, 'fResident': 5, 'gParking': 6}

print(classDict['aGrass'])
plt.figure(figsize=(24, 24))

for i, (k, v) in enumerate(classDict.items()):

    plt.subplot(len(classDict), 1, i+1)

    subroot = os.path.join(root, k)

    imgPath = os.path.join(subroot, os.listdir(subroot)[0])

    img = cv.imread(imgPath)

    plt.imshow(img)

    plt.title(k)
# Checking GPU is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('Training on CPU...')

else:

    print('Training on GPU...')
# Dataset responsible for manipulating data for training as well as training tests.

class DatasetRS(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):

        self.imglist = glob(f'{root}/*/*.jpg')

        self.transform = transform

        

    def __len__(self):

        return len(sself.imglist)

    

    def __getitem__(self, index):

        image = cv.imread(self.imglist[index])

        image = cv.resize(image, dsize=(224, 224))

        className = os.path.split(self.imglist[index])[-2].split('/')[-1]

#         print(className)

        label = classDict[className] 

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 4

VALID_SIZE = 0.5 # percentage of data for validation



#preprocess data

transform_train = transforms.Compose([

    transforms.ToPILImage(),

   # transforms.RandomRotation(0, 0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



transform_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



# Importing data that will be used for training and validation

dataPath = "../input/rsscn7/RSSCN7-master"



# Creating datasets for training and validation

train_data = DatasetRS(dataPath, transform=transform_train)

valid_data = DatasetRS(dataPath, transform=transform_valid)



# Shuffling data and choosing data that will be used for training and validation

num_train = len(glob(f'{dataPath}/*/*.jpg'))

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(VALID_SIZE * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)



print(f"Length train: {len(train_idx)}")

print(f"Length valid: {len(valid_idx)}")
# class Net(nn.Module):

#     def __init__(self, in_channel=3, num_classes=7):

#         super(Net, self).__init__()

        

#         self.conv1 = nn.Sequential(

#             nn.Conv2d(in_channel, 32, 3, padding=1),

#             nn.ReLU(),

#             nn.BatchNorm2d(32),

#             nn.Conv2d(32, 32, 3, stride=2, padding=1),

#             nn.ReLU(),

#             nn.BatchNorm2d(32),

#             nn.MaxPool2d(2, 2),

#             nn.Dropout(0.25)

#         )

        

#         self.conv2 = nn.Sequential(

#             nn.Conv2d(32, 64, 3, padding=1),

#             nn.ReLU(),

#             nn.BatchNorm2d(64),

#             nn.Conv2d(64, 64, 3, stride=2, padding=1),

#             nn.ReLU(),

#             nn.BatchNorm2d(64),

#             nn.MaxPool2d(2, 2),

#             nn.Dropout(0.25)

#         )

        

#         self.conv3 = nn.Sequential(

#             nn.Conv2d(64, 128, 3, padding=1),

#             nn.ReLU(),

#             nn.BatchNorm2d(128),

#             nn.MaxPool2d(2, 2),

#             nn.Dropout(0.25)

#         )



#         self.avgpool = nn.AvgPool2d(kernel_size=4)

        

#         self.fc = nn.Sequential(

#             nn.Linear(1152, num_classes),

#         )

                

        

#     def forward(self, x):

#         x = self.conv1(x)

# #         print("x1.size:", x.size())

#         x = self.conv2(x)

# #         print("x2.size:", x.size())

#         x = self.conv3(x)

# #         print("x3.size:", x.size())

#         x = self.avgpool(x)

# #         print("x4.size:", x.size())

#         x = x.view(x.size(0), -1)

#         return self.fc(x)
from torchvision import models

num_classes = 7

model = models.vgg16()

model.classifier = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, num_classes),

        )

print(model)



if train_on_gpu:

    model.cuda()
LEARNING_RATE = 0.001680



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
epochs = 1

valid_loss_min = np.Inf

train_losses, valid_losses = [], []

history_accuracy = []



for e in range(1, epochs+1):

    running_loss = 0



    for images, labels in train_loader:

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # Clear the gradients, do this because gradients are accumulated.

        optimizer.zero_grad()

        

        # Forward pass, get our log-probabilities.

        ps = model(images)



        # Calculate the loss with the logps and the labels.

        loss = criterion(ps, labels)

        

        # Turning loss back.

        loss.backward()

        

        # Take an update step and few the new weights.

        optimizer.step()

        

        running_loss += loss.item()

    else:

        valid_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations.

        with torch.no_grad():

            model.eval() # change the network to evaluation mode

            for images, labels in valid_loader:

                if train_on_gpu:

                    images, labels = images.cuda(), labels.cuda()

                # Forward pass, get our log-probabilities.

                #log_ps = model(images)

                ps = model(images)

                

                # Calculating probabilities for each class.

                #ps = torch.exp(log_ps)

                

                # Capturing the class more likely.

                _, top_class = ps.topk(1, dim=1)

                

                # Verifying the prediction with the labels provided.

                equals = top_class == labels.view(*top_class.shape)

                

                valid_loss += criterion(ps, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        model.train() # change the network to training mode

        

        train_losses.append(running_loss/len(train_loader))

        valid_losses.append(valid_loss/len(valid_loader))

        history_accuracy.append(accuracy/len(valid_loader))

        

        network_learned = valid_loss < valid_loss_min



        if e == 1 or e % 5 == 0 or network_learned:

            start = time.strftime("%H:%M:%S")

            print(f"Epoch: {e}/{epochs} | ⏰: {start}",

                  f"Training Loss: {running_loss/len(train_loader):.3f}.. ",

                  f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. ",

                  f"Test Accuracy: {accuracy/len(valid_loader):.3f}")

        

        if network_learned:

            valid_loss_min = valid_loss

            torch.save(model.state_dict(), 'model_mtl_mnist.pt')

            print('Detected network improvement, saving current model')
# Viewing training information

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(history_accuracy, label='Validation Accuracy')

plt.legend(frameon=False)
# Importing trained Network with better loss of validation

model.load_state_dict(torch.load('model_mtl_mnist.pt'))



print(model)