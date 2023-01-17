# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, shutil
import multiprocessing
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
            print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
multiprocessing.cpu_count()
import torch
import torchvision

import pickle
import numpy as np
from skimage import io

from tqdm.notebook import tqdm
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
%matplotlib inline
data_root = '/kaggle/input/simpsons4/train/simpsons_dataset'
class_names = os.listdir(data_root)
train_root = '../train'
val_root = '../val'
for dir in [train_root, val_root]:
    for classs in class_names: os.makedirs(os.path.join(
        dir,
        classs
    ), exist_ok = True)
!ls ../val
for dir_name, dirs, files in tqdm(os.walk(data_root)):
    if len(dirs)!=0:
        continue
    classs = dir_name.split('/')[-1]
    for i,file in enumerate(files):
        to_copy = ''
        if i%10:
            to_copy = os.path.join(train_root,classs, file)
        else:
            to_copy = os.path.join(val_root,classs, file)
        shutil.copy(os.path.join(dir_name, file), to_copy)
!ls ../val/abraham_grampa_simpson
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomVerticalFlip(0.1),
#     transforms.ColorJitter(brightness = 0,
#                            contrast = 0.1,
#                            saturation = 0.05,
#                            hue = 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = ImageFolder(train_root, train_transforms)
val_dataset = ImageFolder(val_root, val_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=64,num_workers=2, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size=64)
len(train_dataset)
%%time
X_b, y_b = next(iter(train_dataloader))
X_b = np.rollaxis(np.array(X_b),1,4)
plt.imshow(X_b[0])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fit_model(model, loss, optimizer, scheduler, num_epoch = 50, 
              train_data = train_dataloader, val_data = val_dataloader, alpha = 0.0005):
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epoch):
        running_loss = 0
        running_acc = 0
        
        for X, y in tqdm(train_data):
            
            model.train()
            
            X = X.to(device)
            y = y.to(device)
            
#             reg_loss = 0
#             for param in model.out_fc.parameters():
#                 reg_loss += torch.norm(param, 1)
            
            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss(preds, y)
            pred_class = preds.argmax(dim = 1)
            
            loss_value.backward()
            optimizer.step()
            
            running_loss+=loss_value.item()
            running_acc+=(pred_class.cpu().numpy() == y.cpu().numpy()).mean()
            
        train_losses.append(running_loss/len(train_data))
        print('train:\nepoch - {},\n loss - {} \n, acc - {}'.format(
               epoch, running_loss/len(train_data), running_acc/len(train_data)
        ))
        
        running_loss = 0
        running_acc = 0
        
        for X, y in tqdm(val_data, ascii=True):
            
            model.eval()
            
            
            X = X.to(device)
            y = y.to(device)
            
            preds = model(X)
            loss_value = loss(preds, y)
            pred_class = preds.argmax(dim = 1)
            
            running_loss+=loss_value.item()
            running_acc+=(pred_class.cpu().numpy() == y.cpu().numpy()).mean()
            
            scheduler.step(loss_value)
            
        val_losses.append(running_loss/len(val_data))
        print('val:\nepoch - {},\n loss - {} \n, acc - {}'.format(
               epoch, running_loss/len(val_data), running_acc/len(val_data)
        ))
    
    return train_losses, val_losses
from torch import nn
from torchvision import models
models.mobilenet_v2(pretrained=False, progress=False)
m.classifier
class SimpsonModel(torch.nn.Module):
    
    def __init__(self):
        
        super(SimpsonModel, self).__init__()
        
        self.base_model = models.mobilenet_v2(pretrained=True, progress=False)
        
        for layer in self.base_model.parameters():
            layer.require_grad = True
        
        self.base_model.classifier = nn.Sequential(
                      nn.Dropout(p=0.2, inplace=False),
                        nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        )
#         self.drop = nn.Dropout(0.2)
#         self.tanh = nn.Tanh()
#         self.out_fc = nn.Linear(in_features=256,
#                                 out_features=len(class_names))
        
    def __call__(self,X):
        X = self.base_model(X)
#         X = self.out_fc(self.tanh(X))
        
        return X
model = SimpsonModel()
model = model.to(device)

loss = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3, amsgrad = True)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-6, amsgrad = True)
tr, val = fit_model(model, loss, optimizer, scheduler)
from sklearn.metrics import f1_score, accuracy_score
def predict_on_images(model, dataloader):
    preds = []
    ys = []
    model.eval()

    for X, y in tqdm(val_dataloader):

                X = X.to(device)
                y = y.to(device)

                pred = model(X)
                pred_class = pred.argmax(dim = 1)

                preds.append(pred_class.cpu().numpy())
                ys.append(y.cpu().numpy())
    
    preds = np.hstack(preds)
    ys = np.hstack(ys)
    
    return preds, ys
preds, ys = predict_on_images(model, val_dataloader)

f1_score(ys, preds, average='macro')
test_dataset = ImageFolder('../input/simpsons4/testset/', val_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=64)
from PIL import Image
test_root = '../input/simpsons4/testset/testset/'
preds = []

for im in tqdm(os.listdir(test_root)):
    X = Image.open(os.path.join(test_root, im))
    X = val_transforms(X)
    X = X[np.newaxis,].to(device)
    pred = model(X).argmax(dim = 1).cpu().numpy()[0]
    preds.append((im, class_names[pred]))
import pandas as pd
class_names.sort()
submission = pd.DataFrame(preds)
submission.columns = ['Id', 'Expected']
submission.set_index('Id', inplace = True)
submission
submission.to_csv('submission.csv')
