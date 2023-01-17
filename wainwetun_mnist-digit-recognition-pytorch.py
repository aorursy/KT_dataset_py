import pandas as pd
import seaborn as sns
import numpy as np
import time
import copy
import csv
import pickle

import torch
import torch.utils.data as data
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_train = train["label"]
X_train = train.drop(labels="label", axis=1)

# del train
X_train_list = X_train.values.tolist()
test_list = test.values.tolist()
print(len(test_list))
g = sns.countplot(Y_train)
Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
# https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]
np.eye(9, dtype='uint8')[0]
num_classes = (np.unique(Y_train).shape)[0]
Y_train = to_categorical(Y_train, num_classes)
def train_model(model, criterion, optimizer, num_epoch=25):
    since = time.time()
    model = model.cuda()
#     best_model = copy.deepcopy(model)
#     best_loss = float('inf')
    
    for epoch in range(num_epoch):
        running_loss = 0
        print('Epoch {}/{}'.format(epoch, num_epoch-1))
        print('-' * 10)
        
        for phase in ['train']:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
            
            for data in dataloaders[phase]:
                inputs, labels = data
#                 print(labels.size())
                inputs, labels = Variable(inputs.cuda().float()), Variable(labels.cuda().float())
                
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
#                     print(outputs.size())
                    labels = labels.reshape(outputs.size())
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss = running_loss + loss.item()
                
            epoch_loss = running_loss/dataloaders_size[phase]
            
            print('{} Loss : {:.4f}'.format(phase, epoch_loss))
        
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:0f}s'.format(time_elapsed//60, time_elapsed%60))
    return model

def evaluate_model(model):
    print('Evaluating')
    model.eval()
    
    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda().float()), Variable(labels.cuda().float())
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_list = predicted.cpu().data.numpy().tolist()
#         row = [i, predicted]
        print("Image ID {} is predicted as {}".format(i, predicted_list))
        with open('submission.csv', 'a') as f:
            j = 1
            for p in predicted_list:
                row = [j+batch_size*i, p]
                writer = csv.writer(f)
                writer.writerow(row)
                j = j+1
                
        f.close()
        
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(64,64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(128,256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(256,128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
#             nn.
#             nn.
#             nn.Linear(128, num_classes)
        )
        self.fcnn = nn.Sequential(
            nn.Linear(128*4*4, 120),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(120, 10),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*4*4)
        x = self.fcnn(x)
        return x
class MNISTDataset(data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform =transform
        self.csv_file = csv_file

    def __getitem__(self, index):
        if self.csv_file=='train.csv':
            image = self.data.iloc[index, 1:].values.astype(np.uint8)
    #         print(len(image))
            image = np.reshape(image, (1,28,28)).transpose()
            label = to_categorical(self.data.iloc[index, 0],10)
        else:
            image = self.data.iloc[index, 0:].values.astype(np.uint8)
    #         print(len(image))
            image = np.reshape(image, (1,28,28)).transpose()
            label = to_categorical(0,10)
#         print(label)
#         print(image.size())
        if self.transform is not None:
            image = self.transform(image)
#         print(label.shape)
        return image, label
    
    def __len__(self):
        return len(self.data)
batch_size = 64
transform = {'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), 'test': transforms.Compose([transforms.ToTensor()])}
target_transform = {'train': transforms.Compose([transforms.ToTensor()]), 'test': transforms.Compose([transforms.ToTensor()])}
image_datasets = {x:MNISTDataset(csv_file=x+'.csv', transform=transform[x])
                 for x in ['train', 'test']}
# {X_train_list, test_list}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=10)
              for x in ['train', 'test']}
dataloaders_size = {x: len(dataloaders[x]) for x in ['train', 'test']}
print(dataloaders_size)

num_epochs = 100

model = myCNN().cuda()
distance = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.00001, betas =(0.9,0.999))

tmodel = train_model(model, distance, optimizer, num_epoch=num_epochs)
pickle.dump(tmodel, open('mnist_model_0811.sav', 'wb'))
evaluate_model(tmodel)
summary(model, (1,28,28))