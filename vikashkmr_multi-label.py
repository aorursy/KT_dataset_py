import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from __future__ import print_function

from __future__ import division

import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

from PIL import Image

import time

import os

import copy

import torch

from torch.utils.data import Dataset, DataLoader

print("PyTorch Version: ",torch.__version__)

print("Torchvision Version: ",torchvision.__version__)
# model = models.vgg16(pretrained=True)

# features = model.features
# features
input_size=224

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomRotation([-30, 30]),

        transforms.Resize((input_size,input_size)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'val': transforms.Compose([

        transforms.Resize((input_size,input_size)),

#         transforms.CenterCrop(input_size),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
data_root='../input/data/'

df=pd.read_csv(data_root+'attributes.csv')

df.head()

df.shape
ind=[]

for i in df['filename']:

    if os.path.exists(os.path.join(data_root,"images",i)):

        ind.append(True)

    else:

        ind.append(False)
df['pattern'].value_counts(),df['neck'].value_counts(),df['sleeve_length'].value_counts(),

# train['pattern'].value_counts(),train['neck'].value_counts(),train['sleeve_length'].value_counts(),

# test['pattern'].value_counts(),test['neck'].value_counts(),test['sleeve_length'].value_counts(),


# df.shape
df=df[ind]

df['neck'].fillna(7,inplace=True)

df['sleeve_length'].fillna(4,inplace=True)

df['pattern'].fillna(10,inplace=True)
df.isnull().sum()
msk = np.random.rand(len(df)) < 0.8

train = df[msk]

test = df[~msk]
data_root='../input/data/'

class MyDataset(Dataset):

    def __init__(self, data_root,data,data_transform):

        self.imgs = [os.path.join(data_root,"images",i) for i in data['filename']]

        self.labels=[(torch.tensor(a),torch.tensor(b),torch.tensor(c)) for a,b,c in zip(data['pattern'],data['neck'],data['sleeve_length'])]

        self.data_transform=data_transform

    def __len__(self):

        return len(self.imgs)



    def __getitem__(self, idx):

        return self.data_transform(Image.open(self.imgs[idx])),torch.tensor(self.labels[idx])

train_dataset=MyDataset(data_root,train,data_transforms['train'])

valid_dataset=MyDataset(data_root,test,data_transforms['val'])
len(train_dataset),len(valid_dataset)
train_dl=DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dl=DataLoader(valid_dataset, batch_size=32, shuffle=True)
def get_classifier(n_classes):

    return nn.Sequential(

            nn.Dropout(),

            nn.Linear(25088, 4096),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, n_classes))

def MultiLabelModel():

    resnet34 = models.resnet34(pretrained=True)

#     modules=list(resnet34.children())[:-2]

#     model=nn.Sequential(*modules)

#     model.cuda()

    class Network(nn.Module):

        def __init__(self):

            super().__init__()

            self.features = nn.Sequential(*list(resnet34.children())[:-2])

            for p in self.features.parameters():

                p.requires_grad = True

            self.classifier1 = get_classifier(11)

            self.classifier2 = get_classifier(8)

            self.classifier3 = get_classifier(5)

#             self.fc = nn.Linear(512 * 7 * 7, 100)

#             self.fc1 = nn.Linear(100, 11)

#             self.fc2 = nn.Linear(100, 8)

#             self.fc3 = nn.Linear(100, 5)

#             self.softmax = torch.nn.Softmax(dim=1)



        def forward(self,x):

            x = self.features(x)

            x = x.view(x.size(0), -1)

            y1 = self.classifier1(x)

            y2 = self.classifier2(x)

            y3 = self.classifier3(x)

#             out = model(x)

# #             h_relu = self.linear1(x).clamp(min=0)

#             out = out.view(-1, 512 * 7 * 7)

#             out = F.relu(self.fc(out))

#             y1 =self.softmax(self.fc1(out))

#             y2 = self.softmax(self.fc2(out))

#             y3 = self.softmax(self.fc3(out))

            return y1,y2,y3

    test = Network()

    return test
m=MultiLabelModel()

m.cuda();
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# x = torch.randn(32, 3, 224, 224).to(device)

# y=m(x)

# y[0].shape,y[1].shape,y[2].shape
# x1=torch.rand((32,3))
# x1[:,0]
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

    since = time.time()



    val_acc_history = []



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    best_loss=999999999



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            # print('Sailabh')

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_loss1 = 0.0

            running_loss2 = 0.0

            running_loss3 = 0.0

            running_corrects = 0

            running_corrects1 = 0

            running_corrects2 = 0

            running_corrects3 = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

#                 print('1st batch')

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss

                    # Special case for inception because in training it has an auxiliary output. In train

                    #   mode we calculate the loss by summing the final output and the auxiliary output

                    #   but in testing we only consider the final output.

                    if is_inception and phase == 'train':

                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)

                        loss1 = criterion(outputs, labels)

                        loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                    else:

                        output1,output2,output3 = model(inputs)

#                         print(labels[:,0])

                        loss1 = criterion(output1, labels[:,0].long())

#                         print(loss1)

                        loss2 = criterion(output2, labels[:,1].long())

                        loss3 = criterion(output3, labels[:,2].long())

                        loss = loss1 + loss2 + loss3

#                         print(loss)

                        

                        

                    _, preds1 = torch.max(output1, 1)

                    _, preds2 = torch.max(output2, 1)

                    _, preds3 = torch.max(output3, 1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss1 += loss1.item() * inputs.size(0)

                running_loss2 += loss2.item() * inputs.size(0)

                running_loss3 += loss3.item() * inputs.size(0)

                running_loss += loss.item() * inputs.size(0)

                running_corrects1 += torch.sum(preds1 == labels[:,0].data)

                running_corrects2 += torch.sum(preds2 == labels[:,1].data)

                running_corrects3 += torch.sum(preds3 == labels[:,2].data)

#                 running_corrects += torch.sum(preds == labels.data)



            epoch_loss1 = running_loss1 / len(dataloaders[phase].dataset)

            epoch_loss2 = running_loss2 / len(dataloaders[phase].dataset)

            epoch_loss3 = running_loss3 / len(dataloaders[phase].dataset)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc1 = running_corrects1.double() / len(dataloaders[phase].dataset)

            epoch_acc2 = running_corrects2.double() / len(dataloaders[phase].dataset)

            epoch_acc3 = running_corrects3.double() / len(dataloaders[phase].dataset)

            



            print('{} Pattern Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss1, epoch_acc1))

            print('{} Neck Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss2, epoch_acc2))

            print('{} Sleeve Length Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss3, epoch_acc3))

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))



            # deep copy the model

            if phase == 'val' and epoch_loss < best_loss:

                print('Saving at {} Epoch'.format(epoch))

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), best_path)

            if phase == 'val':

                val_acc_history.append((epoch_acc1,epoch_acc2,epoch_acc3))



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    print('Best val Acc: {:4f}'.format(best_loss))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, val_acc_history







# model, val_acc_history=train_model(m,data_loaders, criterion, optimizer, num_epochs=num_epochs, is_inception=False)
device


import torch.nn.functional as F

from torch.autograd import Variable



class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()

        self.gamma = gamma

        self.alpha = alpha

        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])

        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

        self.size_average = size_average



    def forward(self, input, target):

        if input.dim()>2:

            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W

            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C

            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1,1)



        logpt = F.log_softmax(input)

        logpt = logpt.gather(1,target)

        logpt = logpt.view(-1)

        pt = Variable(logpt.data.exp())



        if self.alpha is not None:

            if self.alpha.type()!=input.data.type():

                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0,target.data.view(-1))

            logpt = logpt * Variable(at)



        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average: return loss.mean()

        else: return loss.sum()
data_loaders={'train':train_dl,'val':valid_dl}

criterion=FocalLoss()

# criterion=nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(m.parameters(), lr=0.0001)

num_epochs=5

lr_rate=0.00007

best_path='model.pth'
model, val_acc_history=train_model(m,data_loaders, criterion, optimizer, num_epochs=30, is_inception=False)
m
# print(dataset.)aind=0

data_transforms_pred = transforms.Compose([

        transforms.Resize((input_size,input_size)),

        # transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

preds_1=[]

preds_2=[]

preds_3=[]

s_t=time.time()

for ind,name in enumerate(test['filename']):

    if ind%100==0:

        print(ind,time.time()-s_t)

#     ind+=1

    f=os.path.join(data_root,'images',name)

    try:

#         act=get_label(f.split('/')[3])

        img=torch.unsqueeze(data_transforms_pred(Image.open(f)),0)

        img = img.to(device)

        pred=model(img)

        _, preds1 = torch.max(pred[0], 1)

        _, preds2 = torch.max(pred[1], 1)

        _, preds3 = torch.max(pred[2], 1)

        preds_1.append(preds1.tolist()[0])

        preds_2.append(preds2.tolist()[0])

        preds_3.append(preds3.tolist()[0])

        

#         x=softmax(pred).tolist()[0][0]

#         dst='/media/vidooly/myfiles/DeepLearning/ml/projects/user/vikash/Adult_pytorch/data/tested/sailabh_collected/'+str(int(x*10))

#         os.makedirs(dst,exist_ok=True)

#         shutil.copy(f,dst+'/'+str(round(x,3))+','+f.split('/')[-1])

    except Exception as e:

        print(f,e)

#         shutil.move(f,'error_files/'+f.split('/')[-1])
test['neck_pred']=preds_2

test['length_pred']=preds_3

test['pattern_pred']=preds_1
data_transforms_pred = transforms.Compose([

        transforms.Resize((input_size,input_size)),

        # transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

preds_1=[]

preds_2=[]

preds_3=[]

s_t=time.time()

for ind,name in enumerate(train['filename']):

    if ind%100==0:

        print(ind,time.time()-s_t)

#     ind+=1

    f=os.path.join(data_root,'images',name)

    try:

#         act=get_label(f.split('/')[3])

        img=torch.unsqueeze(data_transforms_pred(Image.open(f)),0)

        img = img.to(device)

        pred=model(img)

        _, preds1 = torch.max(pred[0], 1)

        _, preds2 = torch.max(pred[1], 1)

        _, preds3 = torch.max(pred[2], 1)

        preds_1.append(preds1.tolist()[0])

        preds_2.append(preds2.tolist()[0])

        preds_3.append(preds3.tolist()[0])

        

#         x=softmax(pred).tolist()[0][0]

#         dst='/media/vidooly/myfiles/DeepLearning/ml/projects/user/vikash/Adult_pytorch/data/tested/sailabh_collected/'+str(int(x*10))

#         os.makedirs(dst,exist_ok=True)

#         shutil.copy(f,dst+'/'+str(round(x,3))+','+f.split('/')[-1])

    except Exception as e:

        print(f,e)

#         shutil.move(f,'error_files/'+f.split('/')[-1])
train['neck_pred']=preds_2

train['length_pred']=preds_3

train['pattern_pred']=preds_1
test.head()
train.head()
test.to_csv('test_pred.csv')

train.to_csv('train_pred.csv')
a=1
