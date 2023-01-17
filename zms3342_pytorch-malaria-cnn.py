# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torchvision

import torch

from torchvision import datasets, models, transforms, datasets

import matplotlib.pyplot as plt

from torch import nn

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils import data

import random



import os

print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images"))
#train test split 



img_dir = "../input/cell-images-for-detecting-malaria/cell_images/cell_images"

transform=transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])

train_data = datasets.ImageFolder(img_dir,transform=transform)



num_workers = 0



#percentage of test set that will be used for validation

valid_size = 0.2



test_size = 0.1



# obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

valid_split = int(np.floor((valid_size) * num_train))

test_split = int(np.floor((valid_size+test_size) * num_train))

valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]



print(f'val_index: {len(valid_idx)},test_index: {len(test_idx)},train_index: {len(train_idx)}')



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)



# setting up the data loaders

train_loader = torch.utils.data.DataLoader(train_data, batch_size=101,

    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=33, 

    sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(train_data, batch_size=106,

    sampler=test_sampler, num_workers=num_workers)



# target class labels

classes = ['infected', 'uninfected']
#quick viewing of the images just to make sure everything looks right. 



def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))



dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(classes[labels[idx]])
# image shape is (32,32,3) We'll start by trying to build our own model. 

# three conv layers and 2 linear layers



#Architecture
import torch.nn.functional as F

class Net(nn.Module): 

    def __init__(self): 

        super(Net, self).__init__()

        

######### CONVOLUTION ##############



        self.convolution_one = nn.Conv2d(3,16,3, padding=1) #(32,32,3) ==> (16,16,16)

        self.convolution_two = nn.Conv2d(16,32,3, padding=1) #(16,16,16) ==> (8,8,32)    

        self.convolution_three = nn.Conv2d(32,64,3 ,padding=1) #(8,8,32) ==> (4,4,64)

        

######### LINEAR ###########



        self.linear_one = nn.Linear(4*4*64,250)

        self.linear_two =nn.Linear(250,2)

        

######### Dropout + pooling ###########        

        self.dropout = nn.Dropout(0.2)

        self.pooling1 = nn.MaxPool2d(2,2)

        

    def forward(self, x): 

        x = self.pooling1(F.relu(self.convolution_one(x)))

        x = self.pooling1(F.relu(self.convolution_two(x)))

        x = self.pooling1(F.relu(self.convolution_three(x)))

        x = x.view(-1, 4*4*64)

        x = self.dropout(x)

        x = F.relu(self.linear_one(x))

        x = self.linear_two(x)

        return x 

model = Net()

print(model)
import torch.optim as optim



#loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.005)



#pushes everything to gpu

train_on_gpu = torch.cuda.is_available()

cuda = torch.device('cuda')

if train_on_gpu:

    model.to(cuda)
epochs = 5

valid_loss_min = np.Inf



for epoch in range(1,epochs+1): 

    training_loss=0

    validation_loss=0

######## Training Phase #############

    model.train

    for image, label in train_loader:

        #push to gpu

        if train_on_gpu:

            image, label = image.cuda(), label.cuda()

        #set grads = 0 for each train loader

        optimizer.zero_grad()

        

        #feedforward + backprop

        output= model(image)

        loss= criterion(output,label)

        loss.backward()

        optimizer.step()

        training_loss+=loss.item()*image.size(0)

        

######## validation Phase #############

    model.eval()

    for image, label in valid_loader:

        if train_on_gpu:

            image, label = image.cuda(), label.cuda()

        output = model(image)

        loss = criterion(output, label)

        validation_loss+=loss.item()*image.size(0)

    

    training_loss = training_loss / (len(train_loader.dataset))

    validation_loss = validation_loss / (len(valid_loader.dataset))

    

    print( "epoch {}:  trianing_loss = {}, validation_loss={}".format(epoch, training_loss, validation_loss))

    

    # Save the best model 

    if validation_loss <= valid_loss_min: 

        print( ' new best')

        torch.save(model.state_dict(), 'model_malaria.pt')

        valid_loss_min = validation_loss

    

    

        

    
#load best model 

model.load_state_dict(torch.load('model_malaria.pt'))
# track test loss

test_loss = 0.0

class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))



model.eval()

# iterate over test data

for data, target in test_loader:

    # move tensors to GPU if CUDA is available

    if torch.cuda.is_available():

        data, target = data.cuda(), target.cuda()

    # forward pass

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to what the actual value is 

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy 



    

    for i in range(106):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1

    



# average loss for tests 

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
# obtain one batch of test images

dataiter = iter(test_loader)

images, labels = dataiter.next()

images.numpy()



# move model inputs to cuda, if GPU available

if torch.cuda.is_available():

    images = images.cuda()



# get sample outputs

output = model(images)

# convert output probabilities to predicted class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images.cpu()[idx])

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))