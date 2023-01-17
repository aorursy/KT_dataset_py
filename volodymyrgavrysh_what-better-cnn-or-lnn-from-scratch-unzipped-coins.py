import torch.nn as nn

import torch.nn.functional as F

from torch.optim import SGD

from torch.utils.data.dataset import Dataset

from torchvision import transforms as T

from torch.utils.data import DataLoader

import torch

from torch import optim



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import shutil

import cv2

from PIL import Image

import time

from copy import deepcopy

import tarfile
random_seed = 42

torch.backends.cudnn.enDatasetable = False

torch.manual_seed(random_seed)
def copy(SHARE=0.1, path="classification_dataset/train_/", moveto="classification_dataset/test_/"):

    

    '''copy files from one directory to other one with size proportion '''



    files = os.listdir(path) # find list of all files

    random_list = np.random.choice(files, int(len(files) * SHARE), replace=False) # take random files

    random_list.sort() # sort data

    for f in random_list: # iterate and concat file path + file name

        src = path + f

        dst = moveto + f

        shutil.move(src, dst) # move ramdom files to test directory from train

    # check the num of files copied

    number_files = len(os.listdir(moveto))

    print(number_files)



# RUN ONLY ONE TIME on your local machine and set paths before (create folders)

# copy(SHARE=0.2)
# prepare objects 

train_tar = '../input/ .... train_.tar.xz'

test_tar = '../input/ ... test_.tar.xz'



# function to extarct 

def read_tar_to_array(file):

    '''Read tar file and convert jpg to array witj cv2.

    From file name extarct labels and convert into array'''

    # Source https://www.kaggle.com/gbonesso/deep-learning-cnn#Extract-images-and-labels-from-images-in-tar.gz-file

    tar = tarfile.open(file, "r:xz")

    # lists to store

    image_list = []

    label_list = []

    # iterate over tar file

    for tarinfo in tar:

        tar.extract(tarinfo.name) # creare an object with tar info

        if(tarinfo.name[-4:] == '.jpg'): # select only jpg files

            image_list.append(Image.open(tarinfo.name).convert("RGB"))

#             image_list.append(np.array(cv2.imread(tarinfo.name, cv2.IMREAD_COLOR))) # transfrom with cv2 photo fils into arrays

            label_list.append(tarinfo.name.split('_/')[1]) # split lables from file names 



# # You may need it if you run on your local comp 

#         "if(tarinfo.isdir()):

#             os.rmdir(tarinfo.name) # check is name exist

#         else:

#             os.remove(tarinfo.name)"



    tar.close() #close tar

    labels = np.array([i.split('_', 1)[0] for i in label_list]) # slit one more time and covert to array 

    

    return image_list, labels



# convert tar arhive file into PIL obj, extract labels

#test_img, test_labels = read_tar_to_array(test_tar)

#train_img, train_labels = read_tar_to_array(train_tar)



# check the lenght of the train test files

#len(test_img), len(test_labels), len(train_img), len(train_labels)
class BrazillianCoins(Dataset):

    

    def __init__(self, root_dir):

        

        """ Data set class for images with tranforming, resizing and preparing them to feed CNN by Torch lib"""

        

        # tranform img. Could be 128 * 96 (as 460 to 380 pic size)

        self.transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]) 

        # set the root dir path

        self.data = os.listdir(root_dir)

        # save direct root path

        self.root_dir = root_dir

                

    def __getitem__(self, idx):

        # dict for target

#         class_dic = {'5':0, '10':1, '25':2, '50':3, '100':4, '85': 5, '20':6} 

        class_dic = dict(list(zip( list(map(str,list(range(5, 180, 5)))),range(0, 35)))) # generate dict for 35 classes 

        # file by index for getitem

        filename = self.data[idx] 

        # img path for iterate

        img_path = os.path.join(self.root_dir + filename) 

        # tranform img to grayscale

        img = Image.open(img_path).convert("RGB") # ot 'L'

        # get label from filename spit 

        label = filename.split('_')[0] 

        # class selector

        target = class_dic[label]

        # apply transformations

        img = self.transforms(img) 

        

        return img, target

    

    def __len__(self):

        return len(self.data)
# init train instance regression_dataset

train = BrazillianCoins(root_dir='../input/br-coins/classification_dataset/all/') # 3K pictures

test = BrazillianCoins(root_dir='../input/br-coins/regression_sample/regression_sample/') # 900 pictures
(train[0][0]).size(), (test[0][0]).size(), train[0][1]
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated

imshow(train[0][0])
class CNN(nn.Module):

    

    def __init__(self, kernels=[3, 3], poolings=[2, 2], channels=[3, 16, 32], 

               paddings=[2, 2], strides=[2, 2], linear_sizes=[32, 32]):

        """

          Define your NN here.

          Args:

            kernels: default [3, 3]. 

              Defines kernel size for each of two convolutional layers.

              Kernel's width equals its height.

            poolings: default [2, 2].

              Deifnes kernel size for each of two pooling layers.

            channels: default [32, 64]. 

              Defines amount of output channels for each of two convolutional layers.

            padding: default [1, 1]. 

              Defines padding size for each of two convolutional layers.

            strides: default [2, 2]. 

              Defines stride size for each of two convolutional layers.

            linear_sizes: default [32, 32]. 

              Defines layer size for each of fully-connected layers.

        """

        super(CNN, self).__init__()

        self.kernels = kernels

        self.channels = channels

        self.paddings = paddings

        self.strides = strides

        self.max_pooling = poolings

        self.linear_sizes = linear_sizes

        self.relu = F.relu



        """ set CNN structure """

        # set firest 2d layers

        self.layer_2d_0 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], \

                      kernel_size=self.kernels[0], stride=self.strides[0], padding=self.paddings[0])

        # set max pooling

        self.max_1 = nn.MaxPool2d(self.max_pooling[0])

         # set firest 2d layers

        self.layer_2d_1 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], \

                      kernel_size=self.kernels[1], stride=self.strides[1], padding=self.paddings[1])

        # set max pooling

        self.max_2 = nn.MaxPool2d(self.max_pooling[1])

        # set linear layers 

        self.layer_1 = nn.Linear(in_features=2048, out_features=128, bias=True)

        self.layer_2 = nn.Linear(in_features=128, out_features=35, bias=True) # 35 classes

        # flatter layer

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # or we can use .view

        self.dropout = nn.Dropout(0.5) 



        """ Workflow for hight and wigth and channels for img 

        m1 = 128 - 3 +1 +2 / 2 = 64 (64*64, 3 channel)

        max_pool = 64/2 = 32 (32*32, 16 channel)

        m2 = 32 - 3 +1 +2 / 2 = (16*16, 32 channel)

        max_pool = 16/2 = 8

        flattern = [8*8*32] = 2048 neurons

        linear1 = [2048, 128] neurons, size pict

        linear2 = [128, 35] classs

        """



    def forward(self, X):

        """

          Forward propagation of your NN.



          Args:

            X: imput data

          Returns

            outputs: nn's output (logits)

        """

        # First convolution layer

        a = self.max_1(self.relu(self.layer_2d_0(X)))

        # second conv layer

        b = self.max_2(self.relu(self.layer_2d_1(a)))

        # flatter layer before linear layers

        c = b.view(-1, 2048) # or flatter self.flatten(X)

        # dropout

        d = self.dropout(c)

        # first linear layer

        i = self.layer_1(d)

        # dropout

        f = self.dropout(i)

        # second lineat layer

        logits = self.layer_2(f)

        # Softmax probs with logits

        probs = F.softmax(logits, dim=1)

        

        return logits, probs 
# set your batch size 

batch_size = 512



# load train and test into Pytorch format (tensors)

train_ = DataLoader(train, batch_size=batch_size, shuffle=True)

test_ = DataLoader(test, batch_size=batch_size, shuffle=True)



# Create dict for selectors

loaders = {'train' : train_, 'test': test_}

sizes = {'train': len(train), 'test': len(test)}
# our class instance

model = CNN()

# CUDA - for ones who can run on his local machine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transfer model to device

model.to(device)
# test with one image 

test_img = train[0][0] # by index 0

test_img = test_img.unsqueeze(0)



logits, probs = model.forward(test_img)

display('Logits of the last linear layer {} \n with probability {} of class'.format(logits, probs))
class FCNN(nn.Module):

    '''Fully connected NN'''

    

    def __init__(self, hidden_sizes=[64, 64]):

        """

          Define your NN here. 



          Args:

            hidden_sizes: default [64, 64]. 

              Defines layer size for each of hidden layers (layer between input layer

              and output layer).

        """

        super(FCNN, self).__init__()

        self.activation = F.relu

        self.hidden_size = hidden_sizes

        self.input = nn.Linear(in_features=49152, out_features=self.hidden_size[0]) # 128 * 128 * n channel

        self.hidden = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])

        self.hidden_0 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])

        self.output = nn.Linear(in_features=self.hidden_size[1], out_features=35) # 35 classes



    def forward(self, X):

        """

          Forward propagation of your NN.



          Args:

            X: imput data

          Returns

            output: nn's output (logits)

            probs: probabilities of classes by softmax function

        """

    

    def forward(self, X):

        # input # z = W*X + b  

        x = X.view(-1, 49152) # channals * with * lengh

        x = self.input(x)       

        a = self.activation(x) # a = f(z)

        

        # First z1 = W*A + b

        z = self.hidden(a) 

        a = self.activation(z) # a1 = f(z1)

        

        # Second 

        z = self.hidden_0(a)

        a = self.activation(z) # a1 = f(z1)

        

        # out 

        logits = self.output(a) # output = w*a1 + b

        probs = F.softmax(logits, dim=1)

        

        return logits, probs
# intializate model instance 

model_f = FCNN()

# CUDA if you have it

device_f = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set to device other model

model_f.to(device_f)
# take any image

test_img = train[0][0] # by index 0

test_img = test_img.unsqueeze(0)



probs = model_f.forward(test_img)

display('Logits of the last linear layer {} \n with probability {} of class'.format(logits, probs))
def train_model(model, loaders, num_epochs, lr, checkpoint_step=None, eval_step=None): # set our parameters

     

    best_model_ = deepcopy(model.state_dict()) # we make copy of each model settings

    best_acc = 0

    epoch_acc_list = []

    epoch_loss_list = []

    

    # instances of optimazer and criterion

    optimizer = SGD(model.parameters(), lr=lr) # As most of NN we use Gradient Desent

    criterion = nn.CrossEntropyLoss() # It is our loss function, like MSE :)

    

    for epoch in range(num_epochs): # we run over epoch.

        print('Epoch # {}'.format(epoch))

        now = time.time()  # let us calculate the time - so it is starting point

        

        for stage in loaders: # in the begginng we set dict - train and test. now we use them to differ two stage - train and test

            if stage == 'train': # we use Pytorch methods to train

                model.train()

            else:

                model.eval() # and evaluate results with test data

            

            current_loss = 0 # set to zero current (in epoh loss and acuracy)

            current_acuracy = 0

            

            for data, target in loaders[stage]: # for each stage (train, test) we load data with our loaders (see the dicts above)

                data = data.to(device) # send them to devices - data and target values

                targets = target.to(device)

                

                optimizer.zero_grad() # set out initial weight of the model to zero

                

                with torch.set_grad_enabled(stage == 'train'): # check if the stage is Train one

                      

                    outputs, probs = model(data) # calculate the logits and probabilities

                    _, preds = torch.max(outputs, 1) # convert them to value

                    loss = criterion(outputs, targets) # calculate the loss 

                

                    if stage == 'train': # backprop as usually in NN and updating weights

                        loss.backward()

                        optimizer.step()

                        

                    # current acc and loss

                    current_loss += loss.item() * data.size(0)

                    current_acuracy += torch.sum(preds == targets.data)

                    

                # calculate loss and acc for per epoh 

                epoch_loss = current_loss / sizes[stage]

                epoch_acc = current_acuracy.double() / sizes[stage]

                

                # add to list 

                epoch_acc_list.append(epoch_acc)

                epoch_loss_list.append(epoch_loss)

            

            # here we set the # of times to save model

            if epoch % checkpoint_step == 1:

                torch.save(model.state_dict(), 'train_valid_epoch{}.pt'.format(epoch+1))

                

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(stage, epoch_loss, epoch_acc))

            

            # deep copy of the model with higt accuracy and with # step to eval

            # The idea is next - is stage is test:

                # we compare best accuracy (it must grow with each epoch) 

                # and each evalulation step we calculate model perfomance on test set

            #  We save time by this. And it the end we save best model

            if stage == 'test' and epoch_acc > best_acc and epoch % eval_step == 0:

                best_acc = epoch_acc

                best_model_ = deepcopy(model.state_dict())

                

        # time diff

        time_passed = time.time() - now

        print('Training complete in {:.0f}m {:.0f}s'.format(time_passed // 60, time_passed % 60))

        print('Best val Acc: {:4f}'.format(best_acc))

        print('----------------------')

        

        # load best model with its weights

        model.load_state_dict(best_model_)

        

    return model, epoch_loss_list, epoch_acc_list
# run 

# model is CNN, loaders is our data (train and test), lr is learning rate

# checkpoint_step - how often we save model 

# and eval_step is how often we evaluate it on test set

model_cnn = train_model(model, loaders, num_epochs=5, lr=0.01, checkpoint_step=2, eval_step=3)
plt.plot(model_cnn[2], label=['test_loss'])

plt.plot(model_cnn[1], label=['test_accouracy'])

plt.ylabel('Values of loss and accuracy')

plt.title('Loss and accuracy')

plt.legend(['Accuracy', 'Loss'])

plt.show();
# linear NN model 

model_f = train_model(model_f, loaders, num_epochs=5, lr=0.1, checkpoint_step=2, eval_step=3)
plt.plot(model_f[2], label=['test_loss'])

plt.plot(model_f[1], label=['test_accouracy'])

plt.ylabel('Values of loss and accuracy')

plt.title('Loss and accuracy')

plt.legend(['Accuracy', 'Loss'])

plt.show();
def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()

    class_dic = dict(list(zip( list(map(str,list(range(5, 180, 5)))),range(0, 35))))

    

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(loaders['test']):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs, preds = model(inputs)

            _, preds = torch.max(outputs, 1)



            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                ax.set_title('predicted: {}'.format(list(class_dic.keys())[list(class_dic.values()).index(preds[j])]))

                imshow(inputs.cpu().data[j])



                if images_so_far == num_images:

                    model.train(mode=was_training)

                    

    return model.train(mode=was_training)



visualize_model(model_cnn[0])