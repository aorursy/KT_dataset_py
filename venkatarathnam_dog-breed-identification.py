###Importing the necessary libraries###

import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import os

import torch

from torchvision import models, datasets, transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import time

import copy

import glob

from PIL import Image

from sklearn.metrics import classification_report
class DogDS(Dataset):

    def __init__(self, root, phase, transforms, label_file = None):

        self.root = root

        self.phase = phase

        self.transforms = transforms

        self.filenames = glob.glob(os.path.join(os.path.join(root, phase),'*'))

        

        self.len = len(self.filenames)

        if (self.phase == "train"):

            self.labels = pd.read_csv(label_file)

            self.classes = self.labels['breed'].unique().tolist()

    def __getitem__(self, index):

        image = Image.open(self.filenames[index])

        image = self.transforms(image)

        filename = self.filenames[index].split('/')[-1]

        filename_no_ext = filename.split('.')[0]

        breed = str(self.labels[self.labels['id'] == filename_no_ext].breed.values[0])

        label = self.classes.index(breed)

        if (self.phase == 'train'):

            return image, label

        return image

    def __len__(self):

        return self.len
DATA_TRANSFORMS = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
dataset = DogDS('../input/train', 'train', DATA_TRANSFORMS['train'], '../input/labels.csv')

dataset.classes
NUM_CLASSES = len(dataset.classes)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))

valid_size = len(dataset) - train_size

dataset_sizes = {'train': train_size, 'valid': valid_size}

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

dataloaders = {'train': DataLoader(train_dataset, batch_size = 64, shuffle = True),

              'valid': DataLoader(valid_dataset, batch_size = valid_size, shuffle = False)}
label_count_lis = [0]*len(dataset.classes)

for index, row in dataset.labels.iterrows():

    label = dataset.classes.index(row['breed'])

    label_count_lis[label] += 1

print ("Minimum samples per class : {0}".format(min(label_count_lis)))

print ("Maximum samples per class : {0}".format(max(label_count_lis)))
plt.figure()

f, ax = plt.subplots(figsize=(25,10))

plt.bar(np.arange(len(dataset.classes)), label_count_lis)

plt.xticks(np.arange(len(dataset.classes)), np.arange(len(dataset.classes)))

plt.ylabel("No. of samples")

plt.xlabel("Classes")

plt.title('train')

plt.show()
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    losses = {'train': [], 'valid':[]}

    acc = {'train': [], 'valid': []}



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        scheduler.step()

        

        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(DEVICE)

                labels = labels.to(DEVICE)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            acc[phase].append(epoch_acc)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                

            if (phase == 'valid' and epoch + 1 == num_epochs):

                print ("--------------")

                print ("Final Classification Report")

                print ("--------------")

                print (classification_report(preds.cpu(), labels.cpu()))



        print()

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(dataloaders['train']))



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    

    plot_xy(np.arange(num_epochs), [losses['train'], losses['valid']], xlabel = 'Epochs', ylabel = 'Loss', title = 'Loss Plot')

    plot_xy(np.arange(num_epochs), [acc['train'], acc['valid']], xlabel = 'Epochs', ylabel = 'Accuracy', title = 'Accuracy Plot')



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model



def plot_xy(x, y, title="", xlabel="", ylabel=""):

    plt.figure()

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    for i in range(len(y)):

        plt.plot(x, y[i], label = str(i))

    plt.show()
model = models.resnet34(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(dataloaders['train']))
model = train_model(dataloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)