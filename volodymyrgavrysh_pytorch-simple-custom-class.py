import torch

import torchvision

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from torchvision import transforms

from torch.optim import SGD

import torch.optim as optim

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# set the random seed

random_seed = 42

torch.backends.cudnn.enable = False

torch.manual_seed(random_seed)
# set the batch size (could be other size)

batch_size = 512



# load original data from Pytorch 

train_load = MNIST('.', train=True, download=True, 

                                transform=transforms.Compose([transforms.ToTensor()])) # here we transform To Tensor

test_load = MNIST('.', train=False, download=True, 

                                transform=transforms.Compose([transforms.ToTensor()]))



# Use Pytorch loader with batch size 

train_ = DataLoader(train_load, batch_size=batch_size, shuffle=True)

test_ = DataLoader(test_load, batch_size=batch_size, shuffle=True)



# Set simple dic to call 

sizes = {'train' : train_load.data.shape[0], 'test' : test_load.data.shape[0]}

loaders = {'train' : train_, 'test': test_}
train_loader = DataLoader(MNIST('.', train=True, download=True, 

                                transform=transforms.Compose([transforms.ToTensor()])),

                          batch_size=batch_size, shuffle=True)



examples = enumerate(train_loader)

i, (data, target) = next(examples)
%matplotlib inline
import matplotlib.pyplot as plt

figure = plt.figure()

num_of_images = 60

for index in range(1, num_of_images + 1):

    plt.subplot(6, 10, index)

    plt.axis('off')

    plt.imshow(data[index].numpy().squeeze(), cmap='gray_r')
import torch.nn as nn

import torch.nn.functional as F





class NeuralNetwork(nn.Module):



    def __init__(self, activation='sigmoid', hidden_size=256):

        '''Args: activation: {sigmoid, relu, tanh} default sigmoid. 

          Defines which activation function to use.

        hidden_size: default 256 Defines the size of hidden layer.'''

        super(NeuralNetwork, self).__init__()

        

        self.activation = activation

        self.hidden_size = hidden_size

        self.input = nn.Linear(in_features=784, out_features=self.hidden_size)

        self.hidden = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        self.output = nn.Linear(in_features=self.hidden_size, out_features=10)

    

    def forward(self, X):

        x = self.input(X) # z = W*X + b        

        a = self.activation(x) # a = f(z)

        z = self.hidden(a) # z1 = W*A + b

        a = self.activation(z) # a1 = f(z1)

        output = self.output(a) # output = w*a1 + b

        probability = F.softmax(output, dim=1) # use softmax 

        

        return output, probability  
# params 

epoch = 5

learning_rate = 0.1

activation = {'sigmoid':F.logsigmoid, 'relu': F.relu, 'lrelu':F.leaky_relu, 'tanh': F.tanh}



# model with activation funct

model = NeuralNetwork(activation=activation['lrelu']) 



# instances

optimizer = SGD(model.parameters(), lr=learning_rate) # which params we want to update

criterion = nn.CrossEntropyLoss()



# CUDA (if you have)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)
# initial lists

test_loss = []

test_acc = []

train_acc = []

train_loss = []



# start iteration

for i in range(epoch):

    print('Epoch # {}'.format(i))

    # save acc and loss

    current_loss = 0

    current_acc = 0

    # iterate over train  / test

    for state in ['train', 'test']:

        # iterate over loaders

        for data, targets in loaders[state]:

            # tranfer data to device 

            data = data.to(device) 

            target = targets.to(device) 

            # change data to shape with view

            data = data.view(-1, 784)

            # for predict and fit we use diff datasets

            if state == 'train':

            # predict 

                output, probs = model(data)

            else: # we dont need to update  if we predict

                with torch.no_grad():

                    output, probs = model(data)

            # max number of probs

            _, preds = torch.max(probs, 1)

            # set gradients to zero (we always do that before new batch)

            optimizer.zero_grad()

            # loss calculation

            loss = criterion(output, targets) #evarage on minibatch

            # condition of state 

            if state == 'train':

                # metod pytorch to update 

                loss.backward()

                # next step by pytorch

                optimizer.step()

            # culc loss mult on batch size 

            current_loss += loss.item() * data.size(0)

            current_acc += torch.sum(preds == targets.data) # compare how our model predict per evelemnt 

        # calc the loss and train

        epoch_loss_train = current_loss / sizes['train']

        epoch_acc_train = current_acc.double() / sizes['train']

        # for graph data train

        train_loss.append(epoch_loss_train)

        train_acc.append(epoch_acc_train)

        # calc the test

        epoch_loss_test = current_loss / sizes['test']

        epoch_acc_test = current_acc.double() / sizes['test']

        # for graph test

        test_loss.append(epoch_loss_train)

        test_acc.append(epoch_acc_train)

        # print 

        print('Epoch TRAIN loss {} Epoch Accuracy {}'.format(np.round(epoch_loss_train, 2), np.round(epoch_acc_train, 2)))

        print('Epoch TEST loss {} Epoch Accuracy {}'.format(np.round(epoch_loss_test, 2), np.round(epoch_acc_test, 2)))

        print('------------------')
plt.plot(range(1, 11), train_acc , label=['train_accouracy'])

plt.plot(range(1, 11), train_loss, label=['train_loss'])

plt.legend(['train acc', 'train_loss'])

plt.xlabel('Epoch')

plt.title('Train loss and accuracy')

plt.show();
plt.plot(range(1, 11), test_acc , label=['test_accouracy'])

plt.plot(range(1, 11), test_loss, label=['test_loss'])

plt.legend(['test acc', 'test_loss'])

plt.xlabel('Epoch')

plt.title('test loss and accuracy')

plt.show();