import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time
import random
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import random 
seed = 520
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
use_gpu = True
labels = pd.read_csv('../input/train_labels.csv')
sub = pd.read_csv('../input/sample_submission.csv')
train_path = '../input/train/'
test_path = '../input/test/'
print(f'{len(os.listdir("../input/train"))} pictures in train.')
print(f'{len(os.listdir("../input/test"))} pictures in test.')
def mask(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
    
    HSV[:,:,0]=cv2.add(HSV[:,:,0], np.zeros(np.shape(HSV[:,:,0]), dtype=np.uint8), mask=mask)
    HSV[:,:,1]=cv2.add(HSV[:,:,1], np.zeros(np.shape(HSV[:,:,1]), dtype=np.uint8), mask=mask)
    return HSV

fig = plt.figure(figsize=(25, 4))
# display 20 images
train_imgs = os.listdir(train_path)
for idx, img in enumerate(np.random.choice(train_imgs, 10)):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    im = Image.open(train_path + img)
    plt.imshow(im)
    ax = fig.add_subplot(2, 20//2, idx+11, xticks=[], yticks=[])
    im = mask(cv2.imread(train_path + img))
    plt.imshow(im)
    lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
    ax.set_title(f'Label: {lab}')
labels.label.value_counts()
train, val = train_test_split(labels, stratify=labels.label, test_size=0.1)
len(val),len(train)
class MyDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.tif')
        image = mask(cv2.imread(img_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
batch_size = 128

trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
model = models.resnet34(pretrained=True)
#for name,param in model.named_parameters():
#    print("\t",name)
for para in list(model.parameters()):
    para.requires_grad=False
for para in list(model.layer3.parameters()):
    para.requires_grad=True
for para in list(model.layer4.parameters()):
    para.requires_grad=True    
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)
if use_gpu:
    model = model.cuda()
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
num_epochs = 6
early_stopping = 4
criterion = nn.BCEWithLogitsLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.Adam(params_to_update, lr=0.001)
#scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=0.01, step_size=5, mode='triangular2')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

best_val_acc = 0
best_epoch = 0
epoch_since_best = 0

for epoch in range(num_epochs):
    at = time.time()
    scheduler.step()
    model.train()
    train_acc = []
    train_loss = []
    for i, data in enumerate(loader_train):
        if i%10 == 0:
            print('.',end='')
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)        
        loss = criterion(outputs[:,1], labels.float())
        loss.backward()
        optimizer.step()
        a = labels.data.cpu().numpy()
        b = outputs[:,-1].detach().cpu().numpy()
        train_acc.append(roc_auc_score(a, b))
        train_loss.append(loss.item())
            
    model.eval()
    valid_acc = []
    val_loss = []
    for _, data in enumerate(loader_valid):
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:,1], labels.float())
        a = labels.data.cpu().numpy()
        b = outputs[:,-1].detach().cpu().numpy()
        valid_acc.append(roc_auc_score(a, b))
        val_loss.append(loss.item())
    
    print()
    bt = time.time()
    print('[Epoch %d] train loss %.6f train acc %.6f  valid loss %.6f valid acc %.6f  time %.6f' % (
        epoch, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(valid_acc),bt-at))
    
    valid_acc = np.mean(valid_acc)
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        best_epoch = epoch
        epoch_since_best = 0
        print('save model...')
        torch.save(model.state_dict(), 'model.pth')
        print('saved.')
    else:
        epoch_since_best += 1
        
    if epoch_since_best > early_stopping:
        break
            
print('Finished Training')
print('best_epoch: %d, best_val_acc %.6f' % (best_epoch, best_val_acc))
trans_train = transforms.Compose([trainsforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([trainsforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)

optimizer = optim.Adam(params_to_update, lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

for epoch in range(6,10):
    at = time.time()
    scheduler.step()
    model.train()
    train_acc = []
    train_loss = []
    for i, data in enumerate(loader_train):
        if i%10 == 0:
            print('.',end='')
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)        
        loss = criterion(outputs[:,1], labels.float())
        loss.backward()
        optimizer.step()
        a = labels.data.cpu().numpy()
        b = outputs[:,-1].detach().cpu().numpy()
        train_acc.append(roc_auc_score(a, b))
        train_loss.append(loss.item())
            
    model.eval()
    valid_acc = []
    val_loss = []
    for _, data in enumerate(loader_valid):
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:,1], labels.float())
        a = labels.data.cpu().numpy()
        b = outputs[:,-1].detach().cpu().numpy()
        valid_acc.append(roc_auc_score(a, b))
        val_loss.append(loss.item())
    
    print()
    bt = time.time()
    print('[Epoch %d] train loss %.6f train acc %.6f  valid loss %.6f valid acc %.6f  time %.6f' % (
        epoch, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(valid_acc),bt-at))
    
    valid_acc = np.mean(valid_acc)
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        best_epoch = epoch
        epoch_since_best = 0
        print('save model...')
        torch.save(model.state_dict(), 'model.pth')
        print('saved.')
    else:
        epoch_since_best += 1
        
    if epoch_since_best > early_stopping:
        break
            
print('Finished Training')
print('best_epoch: %d, best_val_acc %.6f' % (best_epoch, best_val_acc))
dataset_valid = MyDataset(df_data=sub, data_dir=test_path, transform=trans_valid)
loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)

model.eval()

preds = []
for batch_i, (data, target) in tqdm(enumerate(loader_test)):
    data, target = data.cuda(), target.cuda()
    output = model(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)

sub['label'] = preds
sub.to_csv('sub.csv', index=False)