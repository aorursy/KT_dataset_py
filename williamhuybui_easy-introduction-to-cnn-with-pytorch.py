import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import warnings #Turn off warning

warnings.filterwarnings("ignore")

from IPython.display import Image #For image
#Only work with train set

train = pd.read_csv("../input/digit-recognizer/train.csv", dtype=np.float32,nrows=10000)



#Reduce these sample to 2000 for study purpose (200 image for each number)

#Note that we are not using GPU here so the notebook will break if have more than 4000 sample. 

df=pd.DataFrame({})



for i in range(10):

    df=pd.concat([df,train[train.label==i].head(100)])



print(df.shape)

df.sample(5)
index=2

sample_pic=df.iloc[index,1:].values

plt.imshow(sample_pic.reshape(28,28),cmap='gray')
# Prepare Dataset



# split data into features(pixels) and labels(numbers from 0 to 9)

X_temp = df.loc[:,df.columns != "label"].values/255  # Will significantly increase your score

X=(X_temp-X_temp.mean())/(X_temp.std()+ 1e-8) #Highly reccomended

y = df.label.values



# train test split. Size of train data is 80% and size of test data is 20%. 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state = 420) 



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

X_train_torch = torch.from_numpy(X_train)

y_train_torch = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long



# create feature and targets tensor for test set.

X_test_torch = torch.from_numpy(X_test)

y_test_torch = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long
Image(filename = "../input/neural-network/MLP.png")
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()

mlp.fit(X_train,y_train)

predict=mlp.predict(X_test)

score=(predict==y_test).sum()/len(y_test)
print(score)
import torch

import torch.nn as nn

from torchvision import transforms, models

from torch.autograd import Variable

from torch import nn, optim

import torch.nn.functional as F
# smaller batches means that the number of parameter updates per epoch is greater. 

batch_size = 100 

num_epochs = 200



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(X_train_torch,y_train_torch) #This is like panda DataFrame

test = torch.utils.data.TensorDataset(X_test_torch,y_test_torch)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        # 5 Hidden Layer Network

        self.fc1 = nn.Linear(28*28, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64, 10)

        

        # Dropout module with 0.2 probbability

        self.dropout = nn.Dropout(p=0.2) #This is optional

        # Add softmax on output layer

        self.softmax = F.softmax

        

    def forward(self, x): #max(0,x)

        x = self.dropout(F.relu(self.fc1(x))) #rectified linear unit (ReLU)

        x = self.dropout(F.tanh(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.dropout(F.relu(self.fc4(x)))

        

        x = self.softmax(self.fc5(x),dim=1)

        

        return x
# Instantiate our model

model = Classifier()

# Define our loss function

criterion = nn.NLLLoss()

# Define the optimier

optimizer = optim.Adam(model.parameters(), lr=0.0015)

num_display=10

display_epoch=np.append(np.arange(0,num_epochs,int(num_epochs/num_display)), num_epochs-1)



for epoch in range(num_epochs):

    for images, labels in train_loader: # #Images=Total num image/batch_size

        

        optimizer.zero_grad() # Prevent accumulation of gradients

        # Make predictions

        prediction = model(images.float()) #pass set of images to our Classifier(), stored as x and return prediction

        loss = criterion(prediction, labels)#pass in loss function define above

        

        #backprop

        loss.backward() #The neural net store everystep of the calculation. This simply reserves the previous gradient

        optimizer.step() #... Does the update on weight and bias

        

        #################Report####################

    with torch.no_grad():

        model.eval()

        if epoch in display_epoch:

            

            outputs = model(X_test_torch)

            prediction=torch.max(outputs.data,1)[1]



            accuracy = np.array((np.array(prediction) == y_test)).sum()/100

            print('Epoch: {}   Accuracy: {} %'.format(epoch+1, round(accuracy,3)))



Image(filename = "../input/neural-network/CNN.jpeg")
Image(filename = "../input/neural-network/kernel.gif")
Image(filename = "../input/neural-network/maxpool_animation.gif")
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable
# Create CNN Model

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        

        # Convolution 1. DimIn 28*28, Dimout 24*24

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1)

        self.relu1 = nn.ReLU()

        

        # Max pool 1. DimIn 24 * 24, DimOut 12*12

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

     

        # Convolution 2. DimIn 12*12, DimOut 10*10

        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, stride=1, padding=1)

        self.relu2 = nn.ReLU()

        

        # Convolution 3 DimIn 10*10, DimOut 8*8 

        self.cnn3 = nn.Conv2d(in_channels=18, out_channels=54, kernel_size=5, stride=1, padding=1)

        self.relu3 = nn.ReLU()

        

        # Max pool 3: DimIn 8*8 , DimOut 4*4

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        

        # Fully connected 1

        self.fc1 = nn.Linear(54 * 4 * 4, 27*2) 

        

        # Fully connected 2

        self.fc2 = nn.Linear(27*2, 10)

        

        #Dropout function

        self.dropout = nn.Dropout(p=0.2) #This is optional

        # Add softmax on output layer

        self.softmax = F.softmax

        

    def forward(self, x):

        # Convolution 1

        x = self.cnn1(x)

        x = self.relu1(x)

        

        # Max pool 1

        x = self.maxpool1(x)

        

        # Convolution 2 

        x = self.cnn2(x)

        x = self.relu2(x)

        

        # Convolution 3 

        x = self.cnn3(x)

        x = self.relu3(x)

        

        # Max pool 3

        x = self.maxpool3(x)

        x = x.view(x.size(0), -1) #Flatten it before going to FC layer



        # Linear function (readout)

        x = self.dropout(F.relu(self.fc1(x)))

        

        # x = self.fc1(x)

        x = self.softmax(self.fc2(x), dim=1)

        

        return x
# batch_size, epoch and iteration

batch_size = 100

num_epochs = 200



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(X_train_torch,y_train_torch)

test = torch.utils.data.TensorDataset(X_test_torch,y_test_torch)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
# Create CNN

model = CNNModel()



# Cross Entropy Loss 

criterion = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# CNN model training

num_display=10

display_epoch=np.append(np.arange(0,num_epochs,int(num_epochs/num_display)), num_epochs-1)



for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        

        train = Variable(images.view(batch_size,1,28,28))

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and ross entropy loss

        loss = criterion(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        

    with torch.no_grad():

        model.eval()

        if epoch in display_epoch:

            X_test_resize = Variable(X_test_torch.view(X_test_torch.shape[0],1,28,28))

            outputs = model(X_test_resize)

            prediction=torch.max(outputs.data,1)[1]

            accuracy = np.array((prediction == y_test_torch)).sum()/100

            print('Epoch: {}   Accuracy: {} %'.format(epoch+1, round(accuracy,3)))