# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

from pathlib import Path

import torch

from torch.utils.data import TensorDataset ,DataLoader

from torch import nn,optim

import torch.nn.functional as F

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

PATH=Path("../input/")

print(os.listdir("../input/"))
train=pd.read_csv(PATH/'train.csv')

test=pd.read_csv(PATH/'test.csv')

train.shape,test.shape
x=train.drop("label",axis=1)

y=np.array(train['label'])
torch_X_train = torch.from_numpy(x.values).type(torch.FloatTensor)/255

torch_y_train = torch.from_numpy(y).type(torch.LongTensor)

myDataset = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

valid_no  = int(0.2 * len(myDataset))

# so divide the data into trainset and testset

trainSet,testSet = torch.utils.data.random_split(myDataset,(len(myDataset)-valid_no,valid_no))

print(f"len of trainSet {len(trainSet)} , len of testSet {len(testSet)}")

batch_size=64

train_loader  = DataLoader(trainSet , batch_size=batch_size ,shuffle=True) 

test_loader  = DataLoader(testSet , batch_size=batch_size ,shuffle=True)
from torchvision import datasets, transforms

class Network(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)



        # Dropout module with 0.2 drop probability

        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))



        # output so no dropout here

        x = F.log_softmax(self.fc4(x), dim=1)



        return x

        

model=Network()

optimizer=optim.Adam(model.parameters(),lr=0.01)

criterion=nn.NLLLoss()
epochs=2

train_losses,test_losses=[],[]

for e in range(epochs):

    running_loss=0

    for images,labels in train_loader:

        optimizer.zero_grad()

        log_ps=model(images)

        loss=criterion(log_ps,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        

    else:

        test_loss=0

        accuracy=0

        

        with torch.no_grad():

            model.eval()

            for images,labels in test_loader:

                log_ps=model(images)

                test_loss+=criterion(log_ps,labels)

                ps=torch.exp(log_ps)

                top_p,top_class=ps.topk(1,dim=1)

                equals=top_class==labels.view(*top_class.shape)

                accuracy+=torch.mean(equals.type(torch.FloatTensor))

        model.train()

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))   
print("Our model: \n\n", model, '\n')

print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

print(state_dict.keys())

model.load_state_dict(state_dict)
checkpoint = {'input_size': 784,

              'output_size': 10,

              'hidden_layers': [256,128,64],

              'state_dict': model.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
test_images = pd.read_csv("../input/test.csv")

test_image = test_images.loc[:,test_images.columns != "label"].values

test_dataset = torch.from_numpy(test_image).type(torch.FloatTensor)/255

print(test_dataset.shape)

#test_dataset = torch.utils.data.TensorDataset(test_dataset)

new_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle = False)
results = []

with torch.no_grad():

    model.eval()

    for images in new_test_loader:

        output = model(images)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim = 1)

        results += top_class.numpy().tolist()
predictions = np.array(results).flatten()

print(predictions[:5])

print(predictions.shape)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("my_submissions.csv", index=False, header=True)