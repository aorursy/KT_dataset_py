# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Step-1: # Imporing Necessary Libraries



import torch 

import numpy as np

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import os
# Step-2: Loading the datasets from the folder, and converting the data to greyscale and resizing the data



# Note-1: transforms.compose: Transforms are common image transformations. Multiple transforms can be done with "compose".

# Note-2: Channels = 1 is for grayscale image and channels = 3 is for RGB image



transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(255), 

            transforms.CenterCrop(224), transforms.ToTensor()])



train_dataset = datasets.ImageFolder('../input/malaria/Malaria/train', transform=transform)



test_dataset = datasets.ImageFolder('../input/malaria/Malaria/test', transform=transform)
# Step-3: Train loader to make iterator for the images



trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
trainloader
images, labels = next(iter(trainloader))

images.shape
images.view(32, -1).shape
# To view the images



plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');
# Step-4: Defining the model - Multilayer Perceptron with 3 hidden layers, two classes



class classifier(nn.Module):

    def __init__(self):

        super().__init__()

        

        ## Defining Layers from Input to hidden

        self.fc1 = nn.Linear(50176, 10000)

        self.fc2 = nn.Linear(10000, 5000)

        self.fc3 = nn.Linear(5000,1500)

        self.fc4 = nn.Linear(1500,500)

        self.fc5 = nn.Linear(500,100)

        

        ## Output Layer

        self.fc6 = nn.Linear(100, 2)

        

        #Dropout

        self.dropout = nn.Dropout(p=0.02)

        

    def forward(self, y):

        # make sure input tensor is flattened

        y = y.view(y.shape[0], -1)

        

        # Now with dropout

        y = self.dropout(torch.sigmoid(self.fc1(y)))

        y = self.dropout(torch.sigmoid(self.fc2(y)))

        y = self.dropout(torch.sigmoid(self.fc3(y)))

        y = self.dropout(torch.sigmoid(self.fc4(y)))

        y = self.dropout(torch.sigmoid(self.fc5(y)))



        # output: Here, can't apply dropout

        y = F.log_softmax(self.fc6(y), dim=1)



        return y
# Running the model on GPU, finding out the Training loss, Test Loss and the Accuracy



torch.manual_seed(100)

model = classifier()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to('cuda')

epochs = 20

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:

                images, labels = images.to('cuda'), labels.to('cuda')

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))