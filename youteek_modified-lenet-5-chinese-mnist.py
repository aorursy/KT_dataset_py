import torch

import torchvision

from PIL import Image

import torch.nn

import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
torch.cuda.is_available()
%matplotlib inline



for i in range(0, 4):

    image = Image.open('../input/mnist-but-chinese/MNIST_Chinese_Hackathon/Training_Data/' + str(random.randint(1000*i, 1000*(i+2))))

    print(image.format)

    arr = np.array(image)

    print(arr.shape)

    plt.imshow(arr)

    plt.show()

    print(arr.max(), ', ', i)
train = pd.read_csv('../input/mnist-but-chinese/MNIST_Chinese_Hackathon/train.csv')

test = pd.read_csv('../input/mnist-but-chinese/MNIST_Chinese_Hackathon/test.csv')
train.code.value_counts()
X_train = []



for i in train.id:

    img = Image.open('../input/mnist-but-chinese/MNIST_Chinese_Hackathon/Training_Data/' + str(i))

    img_n = np.asarray(img)

    img_n = img_n/255

    img_n = (img_n - 0.5)/0.5

    X_train.append(img_n)    



y_train = train.code
X_train = np.array(X_train)

X_train = X_train.astype(np.float32)



y_train = np.array(y_train)

y_train = y_train.astype(np.float32)

y_train = y_train - 1 # to make the labels go from 0-14, to be consistent with Python's indexing convention for the softmax fn



print('Train set dims = ', (X_train.shape))
plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1), plt.imshow(X_train[random.randint(1, 2500)], cmap='gray')

plt.subplot(2, 2, 2), plt.imshow(X_train[random.randint(2501, 5000)], cmap='gray')

plt.subplot(2, 2, 3), plt.imshow(X_train[random.randint(5001, 7500)], cmap='gray')

plt.subplot(2, 2, 4), plt.imshow(X_train[random.randint(7501, 10000)], cmap='gray')
X_train.shape
from torch import nn

from torch import optim
class Net(nn.Module):   

    def __init__(self):

        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(



            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),

#             nn.LeakyReLU(0.01),

            nn.ReLU(inplace = True),

            

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=0),

#             nn.LeakyReLU(0.01),

            nn.ReLU(inplace = True),    

            nn.MaxPool2d(kernel_size=2, stride=2),

      

            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),

#             nn.LeakyReLU(0.01),

            nn.ReLU(inplace = True),  

            nn.MaxPool2d(kernel_size=2, stride=2) 

      )



        self.FCL = nn.Sequential(

            nn.Linear(3136, 500),

#             nn.LeakyReLU(0.01),

            nn.ReLU(inplace = True),

            nn.Linear(500, 90),

#             nn.LeakyReLU(0.01),

            nn.ReLU(inplace = True),

            nn.Linear(90, 15)

#             nn.Softmax(dim = 1)

      )



    def forward(self, x):

        x = self.conv_layers(x)

        x = x.reshape(x.size(0), -1)

        x = self.FCL(x)

        return x
model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()



if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()

    

print(model)
X_train = X_train.reshape(10000, 1, 64, 64)

X_train  = torch.from_numpy(X_train)

y_train = y_train.astype(int)

y_train = torch.from_numpy(y_train)
losses = []

runs = []



for epoch in range(50):

    running_loss = 0

    for i in range(len(X_train)):



        if torch.cuda.is_available():

            X_train[i] = X_train[i].cuda()

            y_train[i] = y_train[i].cuda()



        optimizer.zero_grad()

        X_train[i] = X_train[i].unsqueeze_(0)

        y_train[i] = y_train[i].unsqueeze_(0)

        output = model(X_train[i][None, ...].cuda()) 



        loss = criterion(output, y_train[[i]].long().cuda())     

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        if i % 10000 == 9999:

            print('[%d, %5d] loss: %.7f' %

                  (epoch + 1, i + 1, running_loss / 10000))

            losses.append(running_loss/10000)

            runs.append(epoch)

            running_loss = 0.0

            

#     if losses[epoch] > losses[epoch - 1]:

#         print("Loss value increased at epoch ", epoch + 1, 

#               "! The least value of loss so far is ", losses[epoch - 1], " and current value is ", losses[epoch])

#         break
import copy



runs_mod = copy.deepcopy(runs)

for i in range(len(runs)):

    runs_mod[i] = runs[i] + 1



plt.plot(runs_mod, losses)

plt.xlabel("Epochs")

plt.ylabel("CrossEntropy Loss")

plt.show()
X_test = []



for i in test.id:

    img = Image.open('../input/mnist-but-chinese/MNIST_Chinese_Hackathon/Testing_Data/' + str(i))

    img_n = np.asarray(img)

    img_n = img_n/255

    img_n = (img_n - 0.5)/0.5

    X_test.append(img_n)



X_test = np.array(X_test)

X_test = X_test.astype(np.float32)

X_test = X_test.reshape(5000, 1, 64, 64)

X_test  = torch.from_numpy(X_test)
test_preds = model(X_test.cuda())
from torch.nn import functional as F



p = F.softmax(test_preds).data
p = p.cpu()
code = []



for i in range(len(p)):

    cache = np.argmax(p[i])

    code.append(cache.item() + 1)
print("The length of the set of predicted values is: ", len(code))

print("\nValue vs. Value Counts:")

print(pd.Series(code).value_counts())
test['code'] = code

test.head(10)
test.code.value_counts()
test.to_csv('./submission.csv', index = False)