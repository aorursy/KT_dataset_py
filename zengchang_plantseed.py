# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train/"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder



print(os.listdir('../input/train'))

encoder = LabelEncoder()

encoder.fit(os.listdir('../input/train'))

print(encoder.transform(os.listdir('../input/train')))
from PIL import Image

import torch

import torch.nn as nn

import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

TRAIN = '../input/train'

TEST = '../input/test'



image = Image.open(TRAIN + '/' + 'Fat Hen' + '/0a1480ed8.png')

image = np.array(image)

print(image.shape)
def create_manifest():

    log = []

    for label, plant in tqdm(zip(encoder.transform(os.listdir(TRAIN)), os.listdir(TRAIN))):

        plant_dir = os.path.join(TRAIN, plant)

        for img in os.listdir(plant_dir):

            img_path = os.path.join(plant_dir, img)

            log.append((label, plant, img_path))

    return log



log = create_manifest()

transform = transforms.Compose([transforms.Resize((144,144)), 

                                transforms.RandomHorizontalFlip(0.5), 

                                transforms.RandomVerticalFlip(0.5), 

                                transforms.ToTensor(), 

                                transforms.Normalize((0.5,),(0.5,))])
log
from sklearn.model_selection import train_test_split



train, val = train_test_split(log, test_size = 0.1, random_state = 29)



class Trainset(Dataset):

    def __init__(self, log, transform = transform):

        self.log = log

    

    def __len__(self):

        return len(self.log)

    

    def __getitem__(self, idx):

        label, _, img_path = self.log[idx]

        img = Image.open(img_path).convert("RGB")

        if transforms:

            img = transform(img)

        return img, label

        

    

class Testset(Dataset):

    def __init__(self, transform = transform):

        self.test = os.listdir(TEST)

    

    def __len__(self):

        return len(self.test)

    

    def __getitem__(self, idx):

        img = Image.open(os.path.join(TEST, self.test[idx])).convert("RGB")

        if transform:

            img = transform(img)

        return img

    

trainset = Trainset(train)

trainset[0][0].dtype

valset = Trainset(val)

print(valset[0][0].shape)

testset = Testset()

testset[0].shape
device = torch.device('cuda')

!pip install torchsummary

from torchsummary import summary
import re

pattern = re.compile('layer4.*')
def train(model, trainloader, optimizer, loss_fn, epoch):

    sum_loss = 0

    correct = 0

    total = 0

    model.train()

    #progress_bar = tqdm(enumerate(trainloader))

    for idx, (feature, label) in enumerate(trainloader):

        feature = feature.to(device)

        label = label.to(device)

        

        optimizer.zero_grad()

        out = model(feature)

        _, pred = torch.max(out, dim = 1)

        correct += (label == pred).sum().item()

        total += len(feature)

        loss = loss_fn(out, label)

        sum_loss += loss.item()

        loss.backward()

        optimizer.step()

        

        print('\rTrain {:3d} [{:2d}/{:2d} ({:.4%})] Loss: {:.4f} Acc: {:.4%}'.format(epoch, idx + 1, 

                                                                                                          len(trainloader), 

                                                                                                          (idx + 1) / len(trainloader), 

                                                                                                          sum_loss / (idx + 1),

                                                                                                          correct / total), end = '')

    print()

    

def val(model, valloader, loss_fn, epoch):

    correct = 0

    total = 0

    sum_loss = 0

    model.eval()

    with torch.no_grad():

        for idx, (feature, label) in enumerate(valloader):

            feature = feature.to(device)

            label = label.to(device)



            out = model(feature)

            _, pred = torch.max(out, dim = 1)

            correct += (label == pred).sum().item()

            total += len(feature)

            loss = loss_fn(out, label)

            sum_loss += loss.item()

        print('Val {:3d} Loss: {:.4f} Acc: {:.4%}'.format(epoch, sum_loss / len(valloader), correct / total))

    return correct / total



def test(model, testloader):

    results = []

    model.eval()

    with torch.no_grad():

        for feature in testloader:

            feature = feature.to(device)

            out = model(feature)

            _, pred = torch.max(out, dim = 1)

            results.extend(pred.cpu().detach().numpy().tolist())

    return results

        

class Net(nn.Module):

    def __init__(self, num_classes):

        super(Net, self).__init__()

        self.model = models.resnet18(pretrained = True)

        for name, parameter in self.model.named_parameters():

            if not (re.match(pattern, name) or name == 'fc'):

                parameter.requires_grad = False

        self.model.fc = nn.Linear(512, num_classes)

    

    def forward(self, x):

        x = self.model(x)

        return x

        

def main():

    #model = models.resnet.ResNet(models.resnet.BasicBlock, [1,1,1,1])

    #model = models.resnet18(pretrained=True)

    '''

    for parameter in model.parameters():

        parameter.requires_grad = False

    model.fc = nn.Linear(512, 12)

    '''

    model = Net(12)

    model.to(device)

    summary(model, (3,144,144))

    

    trainloader = DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 4)

    valloader = DataLoader(valset, batch_size = 64, shuffle = False, num_workers = 4)

    testloader = DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 4)

    

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.01, betas = (0.9,0.999), weight_decay = 0.00001)

    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.01, momentum = 0.9, weight_decay = 0.00001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.3, last_epoch = -1)

    

    import copy

    

    best_acc = 0

    model_duplicated = None

    for epoch in range(50):

        results = test(model, testloader)

        train(model, trainloader, optimizer, loss_fn, epoch + 1)

        val_acc = val(model, valloader, loss_fn, epoch + 1)

        if val_acc >= best_acc:

            print('save model')

            best_acc = val_acc

            model_duplicated = copy.deepcopy(model)

    model = model_duplicated

    results = test(model, testloader)

    return model, results
model, results = main()
predictions = encoder.inverse_transform(results)
imgs = np.array(testset.test)

submission = pd.DataFrame({'file':testset.test, 'species':predictions})
submission.to_csv('submission.csv', index = None)