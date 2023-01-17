%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, glob, time, copy, random, zipfile

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm





import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

import torch.nn.functional as F

import torchvision

from torchvision import models, transforms
torch.__version__
vgg16path='../input/vgg-16/vgg16-397923af.pth'

vgg16bnpath='../input/vgg-16/vgg16_bn-6c64b313.pth'

vgg19path='../input/vgg-19/vgg19-dcbb9e9d.pth'

vgg19bnpath='../input/vgg-19bn/vgg19_bn-c79401a0.pth'

os.listdir('../input/vgg-19')
# Check Current Directory

os.listdir('../input/dogs-vs-cats-redux-kernels-edition')
# Make Directory for extracting from Zip

os.makedirs('../data', exist_ok=True)
# Train_dir, Test_dir

base_dir = '../input/dogs-vs-cats-redux-kernels-edition'

train_dir = '../data/train'

test_dir = '../data/test'
# Extract All Data From Zip to "../data" Directory

with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:

    train_zip.extractall('../data')

    

with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:

    test_zip.extractall('../data')
# Check File Name

os.listdir(train_dir)[:5]
# FilePath List

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))

test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
img = Image.open(train_list[0])

plt.imshow(img)

plt.axis('off')

plt.show()
img = Image.open(test_list[0])

plt.imshow(img)

plt.axis('off')

plt.show()
# Label is contained in filepath

train_list[:5]
# Image_Id is contained in filepath

test_list[:5]
# Get Label

train_list[0].split('/')[-1].split('.')[0]
# Get Image_Id

int(test_list[0].split('/')[-1].split('.')[0])
# Number of Train Image

len(train_list)
# Nuber of Test Image

len(test_list)
# Divide Train, Valid Data

train_list, val_list = train_test_split(train_list, test_size=0.1)
print(len(train_list))

print(len(val_list))
# Data Augumentation

class ImageTransform():

    

    def __init__(self, resize, mean, std):

        self.data_transform = {

            'train': transforms.Compose([

                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ]),

            'val': transforms.Compose([

                transforms.Resize(256),

                transforms.CenterCrop(resize),

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ])

        }

        

    def __call__(self, img, phase):

        return self.data_transform[phase](img)
# Dataset

class DogvsCatDataset(data.Dataset):

    

    def __init__(self, file_list, transform=None, phase='train'):    

        self.file_list = file_list

        self.transform = transform

        self.phase = phase

        

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self, idx):

        

        img_path = self.file_list[idx]

        img = Image.open(img_path)

        

        img_transformed = self.transform(img, self.phase)

        

        # Get Label

        label = img_path.split('/')[-1].split('.')[0]

        if label == 'dog':

            label = 1

        elif label == 'cat':

            label = 0



        return img_transformed, label
# Config

size = 224

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataset

train_dataset = DogvsCatDataset(train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = DogvsCatDataset(val_list, transform=ImageTransform(size, mean, std), phase='val')



# Operation Check

print('Operation Check')

index = 0

print(train_dataset.__getitem__(index)[0].size())

print(train_dataset.__getitem__(index)[1])
# DataLoader

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}



# Operation Check

print('Operation Check')

batch_iterator = iter(train_dataloader)

inputs, label = next(batch_iterator)

print(inputs.size())

print(label)
# VGG16 Model Loading

use_pretrained = False

#net = models.vgg16(pretrained=use_pretrained)

#pre=torch.load(vgg16path)

#net.load_state_dict(pre)



#net = models.resnet101(True)

net = models.resnet152(True)

#net = models.resnet50(True)

class_num = 2 #假设要分类数目是200

channel_in = net.fc.in_features#获取fc层的输入通道数

#然后把resnet-101的fc层替换成300类别的fc层

#net = models.densenet121(pretrained=True)

#net.classifier=nn.Linear(1024,2)

net.fc = nn.Linear(channel_in,class_num)

print(net)



# Change Last Layer

# Output Features 1000 → 2

#net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

print('Done')
# Specify The Layers for updating

params_to_update = []



#update_params_name = ['classifier.6.weight', 'classifier.6.bias']  #vgg

update_params_name = ['fc.weight', 'fc.bias']   #resnet

#update_params_name = ['classifier.weight', 'classifier.bias'] #densenet

for name, param in net.named_parameters():

    print(name,param)

    if name in update_params_name:

        param.requires_grad = True

        params_to_update.append(param)

        print(name)

    else:

        param.requires_grad = False
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):

    

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())

    best_acc = 0.0

    net = net.to(device)

    

    for epoch in range(num_epoch):

        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        print('-'*20)

        

        for phase in ['train', 'val']:

            

            if phase == 'train':

                net.train()

            else:

                net.eval()

                

            epoch_loss = 0.0

            epoch_corrects = 0

            

            for inputs, labels in tqdm(dataloader_dict[phase]):

                inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad()

                

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                        

                    epoch_loss += loss.item() * inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels.data)

                    

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)

            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(net.state_dict())

                

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    net.load_state_dict(best_model_wts)

    return net
# Train

num_epoch = 5

net = train_model(net, dataloader_dict, criterion, optimizer, num_epoch)
# Prediction

id_list = []

pred_list = []



#print(test_list)



with torch.no_grad():

    

    for test_path in tqdm(test_list):

        img = Image.open(test_path)

        #print(img)

        _id = int(test_path.split('/')[-1].split('.')[0])



        transform = ImageTransform(size, mean, std)

        img = transform(img, phase='val')

        img = img.unsqueeze(0)

        img = img.to(device)



        net.eval()



        outputs = net(img)

        #print(F.softmax(outputs, dim=1)[:])

        preds = F.softmax(outputs, dim=1)[:, 1].tolist()

        preds=np.clip(preds,0.005,0.995)

        #print(preds)

        

        id_list.append(_id)

        pred_list.append(preds[0])

    

    

res = pd.DataFrame({

    'id': id_list,

    'label': pred_list

})



res.sort_values(by='id', inplace=True)

res.reset_index(drop=True, inplace=True)



res.to_csv('submission.csv', index=False)
res.head(10)
# Visualize Prediction

id_list = []

class_ = {0: 'cat', 1: 'dog'}



fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')



for ax in axes.ravel():

    

    i = random.choice(res['id'].values)

    

    label = res.loc[res['id'] == i, 'label'].values[0]

    if label > 0.5:

        label = 1

    else:

        label = 0

        

    img_path = os.path.join(test_dir, '{}.jpg'.format(i))

    img = Image.open(img_path)

    

    ax.set_title(class_[label])

    ax.imshow(img)
print('done')