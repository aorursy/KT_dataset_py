import warnings

import csv

import numpy as np

import pandas as pd



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import torch

from torch import np 

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

from torchvision import transforms

import torch.nn as nn

import torch.optim as optim

from torch.autograd import Variable
class DigitDataset(Dataset):



    def __init__(self,csv_path,dtype,mode):

        train_data = pd.read_csv(csv_path)

        self.dtype = dtype

        self.mode=mode

        if(mode=="train" or mode=="val"):

            labels=np.ravel(train_data.ix[:,0:1])

            pixels=train_data.ix[:,1:]

            pixels_train, pixels_test, labels_train, labels_test = train_test_split(pixels, labels, random_state=0,train_size=0.9)

            self.N=pixels_train.shape[0]

            self.V=pixels_test.shape[0]

            self.pixels_train=np.array(pixels_train).reshape([self.N,1,28,28])

            self.labels_train=np.array(labels_train).reshape([self.N,1])

            self.labels_test=np.array(labels_test).reshape([self.V,1])

            self.pixels_test=np.array(pixels_test).reshape([self.V,1,28,28])

        

        if(mode=="test"):

            test_data=pd.read_csv("../input/test.csv")

            self.T=test_data.shape[0]

            self.test=np.array(test_data).reshape([self.T,1,28,28])

   

    def __getitem__(self,index):

        if(self.mode=="train"):

            label=torch.from_numpy(self.labels_train[index]).type(self.dtype)

            img=torch.from_numpy(self.pixels_train[index]).type(self.dtype)

            return img, label

        

        if(self.mode=="val"):

            label=torch.from_numpy(self.labels_test[index]).type(self.dtype)

            img=torch.from_numpy(self.pixels_test[index]).type(self.dtype)

            return img,label

        

        if(self.mode=="test"):

            img=torch.from_numpy(self.test[index]).type(self.dtype)

            return img

    

    def __len__(self):

        if(self.mode=="train"):

            return self.N

        if(self.mode=="val"):

            return self.V

        if(self.mode=="test"):

            return self.T
train_path = '../input/train.csv'

test_path='../input/test.csv'

dtype = torch.FloatTensor



training_dataset = DigitDataset(train_path, dtype,"train")

train_loader = DataLoader(training_dataset,batch_size=256,shuffle=True)



val_dataset=DigitDataset(train_path, dtype,"val")

val_loader = DataLoader(val_dataset)



test_dataset=DigitDataset(test_path,dtype,"test")

test_loader=DataLoader(test_dataset)
class Flatten(nn.Module):

    def forward(self, x):

        N, C, H, W = x.size() # read in N, C, H, W

        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
temp_model=nn.Sequential(

   nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(32),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#1

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(64),

    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(64),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#2

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(128),

    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(128),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#3

    Flatten())



temp_model = temp_model.type(dtype)

temp_model.train()

size=0

for t, (x, y) in enumerate(train_loader):

            x_var = Variable(x.type(dtype))

            size=temp_model(x_var).size()[1]

            if(t==0):

                break
def train(model, loss_fn, optimizer, num_epochs = 1):

    for epoch in range(num_epochs):

        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))

        model.train()

        for t, (x, y) in enumerate(train_loader):

            x_var = Variable(x.type(dtype))

            y_var = Variable(y[:,0].type(dtype).long())



            scores = model(x_var)

            loss = loss_fn(scores, y_var)

            if (t + 1) % 10 == 0:

                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



def check_accuracy(model, loader):  

    num_correct = 0

    num_samples = 0

    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)

    for i, (x, y) in enumerate(loader):

        x_var = Variable(x.type(dtype), volatile=True)

        y=y.numpy()

        scores = model(x_var)

        scores=scores.data.numpy()

        preds = scores.argmax()

        num_correct += (preds == y)

        num_samples += 1

    acc = float(num_correct) / num_samples

    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



def submission(model,loader):

    labels=[]

    for i,x in enumerate(loader):

        if(i%1000==0):

            print("prediction: "+str(i/1000))

        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)

        scores=scores.data.numpy()

        preds = scores.argmax()

        labels.append(preds)

    

    with open('solution.csv', 'w') as csvfile:

        fields = ['ImageId', 'Label']

        writer = csv.DictWriter(csvfile, fieldnames=fields)

        writer.writeheader()

        for i in range(len(labels)):

            if(i%1000==0):

                print("writing: "+str(i/1000))

            writer.writerow({'ImageId': str(i+1), 'Label': str(labels[i])})
model = nn.Sequential(

   nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(32),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#1

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(64),

    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(64),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#2

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(128),

    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(inplace=True),

    nn.BatchNorm2d(128),

    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),#3

    Flatten(),

    nn.Linear(size, 128),

    nn.BatchNorm1d(128),

    nn.ReLU(inplace=True),

    nn.Linear(128, 10))



model = model.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)

optimizer = optim.Adagrad(model.parameters(), lr=2e-2,lr_decay=0.0,weight_decay=0)#2e-2



train(model, loss_fn, optimizer, num_epochs=0)

torch.save(model, "saved_model")

the_model = torch.load("saved_model")

#check_accuracy(model, val_loader)
check_accuracy(the_model, val_loader)
submission(the_model, test_loader)