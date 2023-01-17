import torch

import torchvision

import numpy as np

import pandas as pd

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
import os



# create transforms in this case we will applay Random rotation on our data, then we resize our images to 64 * 64 and all that we will be transferred to a tansors than normalized

images_transform = {'train':transforms.Compose([transforms.RandomRotation(20),

                                                transforms.RandomCrop((16,16)),

                                               

                                                transforms.Resize((64,64)),

                                                transforms.Grayscale(num_output_channels = 1),

                                                transforms.ToTensor(),

                                ]),

                    'test':transforms.Compose([ transforms.Resize((64,64)),

                                                transforms.Grayscale(num_output_channels = 1),

                                                transforms.ToTensor(),

                                ])

                   

                   }



# we save the data dirs in the a dict to use it after



images_dir = { 

                'train': '../input/somefaces/data/train_images',

                'test': '../input/somefaces/data/test_images'

             }





## we load the data and transform it using our transform

images_data = { 'train': torchvision.datasets.ImageFolder(images_dir['train'],transform=images_transform['train']),

                'test': torchvision.datasets.ImageFolder(images_dir['test'],transform=images_transform['test'])

              }



### in our case i will split the training data into 3 subsets  a train, validation and another test set because i think that the test set that we have is not enough .

# first we calculate the data that we have (we won't have the exact number because we did some data augmentation, but it's okay we don't need to have the exact number)

number_train_data = len(os.listdir('../input/somefaces/data/train_images/1')) + len(os.listdir('../input/somefaces/data/train_images/0')) 

# we create a list of indices 

indices = np.arange(number_train_data)

# we shuffle it

np.random.shuffle(indices)



# we calculate the validationset's size and it will be the same size as the test_2 (2nd version of testset)

valid_size = int(0.35 *  number_train_data)



# we create our laoders 

data_loaders = { 'train': torch.utils.data.DataLoader(images_data['train'],batch_size = 32,sampler=SubsetRandomSampler(indices[2 * valid_size:])),

                 'valid': torch.utils.data.DataLoader(images_data['train'],batch_size = 32,sampler=SubsetRandomSampler(indices[:valid_size])),  

                 'test_1': torch.utils.data.DataLoader(images_data['test'],batch_size = 32,shuffle=True),             

                 'test_2': torch.utils.data.DataLoader(images_data['train'],batch_size = 32,sampler=SubsetRandomSampler(indices[valid_size: 2* valid_size]))

               }
def visulize(data):

        plt.subplot(1, 3, 1)

        plt.imshow(data[0][0].squeeze(), cmap='gray')

        plt.subplot(1,3,2)

        plt.imshow(data[0][2].squeeze(), cmap='gray')

        plt.subplot(1,3,3)

        plt.imshow(data[0][3].squeeze(), cmap='gray')
iterate = iter(data_loaders['train'])

visulize(next(iterate))
iterate = iter(data_loaders['test_1'])

visulize(next(iterate))
iterate = iter(data_loaders['test_2'])

visulize(next(iterate))
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        ## ----------------Convolution layer

        self.conv_1 = nn.Conv2d(1,6,kernel_size = 3, stride = 2)

        self.conv_2 = nn.Conv2d(6,6,kernel_size = 3, stride = 2)        

        self.conv_3 = nn.Conv2d(6,3,kernel_size = 3, stride = 2)   

        

        ##---------------- pooling layers

        

        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        # calculate the output from the conv layers

        self.fc_input_size = 3*3*3

        

        ##--------------- fully connected layers

        self.fc1 = nn.Linear(self.fc_input_size,64)

        self.fc2 = nn.Linear(64,128)

        self.fc3 = nn.Linear(128,2)

        ## create a dropout with p=0.15

        self.dropout = nn.Dropout(p = 0.15)

    



    def forward(self, x):

        # conv 1

        x = self.conv_1(x)

        x = F.relu(x)

        # conv 2

        x = self.conv_2(x)

        x = F.relu(x)

        

        x = self.pooling(x)

        # conv 3

        x = self.conv_3(x)

        x = F.relu(x)

        

        ### transform the output of the last layer to a  vector

        

        x = x.view(-1,self.fc_input_size)

        

        

        # fully connnected 1

        x = self.fc1(x)

        x = self.dropout(x)

        x = F.relu(x)

        

        # fully connnected 2

        x = self.fc2(x)

        x = self.dropout(x)

        x = F.relu(x)

        

        

        # output 

        x = self.fc3(x)

        x = F.softmax(x,dim=1)

        return x
# before we define anything we will let our model know in what device we are working on now, if it's GPU (it will be happy if it is), or a simple poor CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def testing(model = None,testloader = None,criterion = None, print_accuracy = False):

    correct = 0

    total = 0

    loss = 0

    more_statistic = {0:{0:0,1:0},1:{0:0,1:0}}

    with torch.no_grad():

        for data in testloader:

            inputs, labels = data



            inputs = inputs.to(device)

            labels = labels.to(device)

            

            result = model(inputs)

            loss += criterion(result,labels)

            _,predicted = torch.max(result.data, 1)

            for i in range(len(predicted)):

                more_statistic[int(labels[i])][int(predicted[i])] += 1

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    if print_accuracy:

        print('Accurcy is :: {:.2f}' .format(100 * correct / total))

    return loss.item() , (100 * correct / total),more_statistic
def trianing(epochs = 100,model = None,criterion = None, optimizer = None, train_loader = None,valid_loader = None,saved_file_name = "checkpoint_v10.pth"):

    train_losses = list()

    valid_losses = list()

    min_loss = 1e6

    total = 0

    counter = 0

    for epoch in range(epochs):

        if counter == 40:

            print('{} epochs without any changes it means we are iin the overfitting phase \n \t bey bey')

            break

        epoch_loss = 0

        for i,data in enumerate(train_loader,1):

           

            

            inputs, labels = data

            

            if epoch == 0:

                total += labels.size(0)

            

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            optimizer.zero_grad()

            result = model(inputs)

            loss = criterion(result,labels)

            loss.backward()

            optimizer.step()

            

            epoch_loss += loss.item()

            

        train_losses.append(epoch_loss)

        validation_loss,_,__ = testing(model = model,testloader = valid_loader,criterion = criterion)

        valid_losses.append(validation_loss)

        counter += 1

        print('Epoch {}\t train loss :: {:.2f} and the validation loss is ::{:.2f}'.format(epoch, epoch_loss,validation_loss))

        if min_loss > validation_loss:

            counter = 0

            min_loss = validation_loss

            print('#------------------  saving the model in {} epoch with validation loss :: {:.2f}'.format(epoch,validation_loss))

            torch.save(model.state_dict(),saved_file_name)

    return train_losses,valid_losses
import torch.optim as optim

model_v1 = Net().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_v1.parameters(),lr = 1e-4)
saved_file_name = "checkpoint_v12.pth"

train_loss,valid_loss = trianing(epochs = 350,

         model = model_v1,

         criterion = criterion,

         optimizer = optimizer, 

         train_loader = data_loaders['train'],

         valid_loader = data_loaders['valid'],

         saved_file_name = saved_file_name)
plt.plot(range(len(valid_loss)),valid_loss,label = "Validation loss")

plt.plot(range(len(train_loss)),train_loss,label = "training loss")

plt.legend(loc='center left')
model_v1.load_state_dict(torch.load(saved_file_name,map_location='cpu'))


def print_statistics(statistics):

    ### printing

    total = statistics[1][1]+statistics[1][0]+statistics[0][1]+statistics[0][0]

    print('\t ----------------------------------')

    print('tested with  :: {} \t accuracy was ::{:.2f}% \t detected positives :: {}/{} \t detected negatives :: {}/{}'.format(total,100 * ((statistics[1][1]+statistics[0][0])/total),statistics[1][1],statistics[1][1]+statistics[1][0],statistics[0][0],statistics[0][0]+statistics[0][1]))

    print('\t ----------------------------------')

    print("\t Confusion Matrix")

    print('True Positive \t {}'.format(statistics[1][1]))

    print('False Negative\t {}'.format(statistics[0][1]))    

    print('False Positive\t {}'.format(statistics[1][0]))    

    print('True Negative \t {}'.format(statistics[0][0]))

    

    print("\t  Precision and Recall")

    print('Precision  {:.3f}\t'.format(statistics[1][1]/(statistics[1][1] + statistics[1][0])))

    print('Recall     {:.3f}\t'.format(statistics[1][1]/(statistics[1][1] + statistics[0][1])))    
loss,accuracy,statistics = testing(model_v1,data_loaders['test_1'],criterion,False)

print_statistics(statistics)
loss,accuracy,statistics = testing(model_v1,data_loaders['test_2'],criterion,False)

print_statistics(statistics)
import cv2

import matplotlib.pyplot as plt

import numpy as np



# read the image and define the stepSize and window size 

# (width,height)





def window_sliding(pic_name = "../input/justpeople/happy_young_people2.jpg",window_size =(64,64),step_size = 10,iterations = 3, pic_reducing = 0.2):

    threshold = 0.8

    model_v1.to('cpu')

    image = cv2.imread(pic_name) # your image path

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h = (image.shape[0] // window_size[0])

    w = (image.shape[1] // window_size[1])

    

    windows =list()

    win__ = list()

    for i in range(iterations):

        

        h = int(h * (1 - i * pic_reducing )) 

        w = int(w * (1 - i * pic_reducing ))

        temp = cv2.resize(image, (w * window_size[1],h * window_size[0]), interpolation = cv2.INTER_LINEAR)

        for x in range(0, image.shape[1]  , step_size):

            for y in range(0, image.shape[0] , step_size):

                window ={"image":temp[x:x + window_size[1], y:y + window_size[0]],

                         "location": ([x,x + window_size[1]], [y,y + window_size[0]]),

                         "pic_reducing": i * pic_reducing

                         }

                # test if it contines a face or not in the same time 

                sub_image =  torch.tensor(window['image'])#.transpose((2,0,1))).unsqueeze(0)

                try:

                    sub_image = transforms.ToPILImage()(sub_image)

                    a = images_transform["test"](sub_image)

                    result = model_v1(a.float().unsqueeze(0))

                    win__.append(window)

                    if result[0,1] > threshold:

                        window['probabily'] = result[0,1]

                        windows.append(window)

                        

                except:

                    pass

    return image,windows,win__
image, boxes,_ = window_sliding()

print(len(boxes))
im = _[297]['image']

plt.subplot(1, 3, 1)

plt.imshow(im )

sub_image =  torch.tensor(im)

sub_image = transforms.ToPILImage()(sub_image)

sub_image = transforms.Grayscale(num_output_channels=1)(sub_image)



a = images_transform["test"](sub_image)

plt.subplot(1, 3, 2)

plt.imshow(a.squeeze(), cmap='gray')

print(a.size())

result = model_v1(a.float().unsqueeze(0))

print(result)
import matplotlib.patches as patches

im = np.array(np.array(image).astype('uint8'))



# Create figure and axes

fig,ax = plt.subplots(1)



# Display the image

ax.imshow(im)



# Create a Rectangle patch

for i in boxes:

    rect = patches.Rectangle((i['location'][1][0] * (1 + i['pic_reducing']) ,

                              i['location'][0][0] * (1+ i['pic_reducing'])),

                             int(64 * (1 + i['pic_reducing'])),

                             int(64 * (1 + i['pic_reducing'])),

                             linewidth=1,edgecolor='r',facecolor='none')

    # add the boxes to the image

    ax.add_patch(rect)



plt.show()
threshold = 0.5

remove = list()

for i in range(len(boxes)):

    for j in range(i+1,len(boxes)):

        if j not in remove:

            y = abs(boxes[i]['location'][1][0] - boxes[j]['location'][1][0])

            x = abs(boxes[i]['location'][0][0] - boxes[j]['location'][0][0])

            result = ((x<=64)*(y<=64)*((64 - x) + (64 - y)))/ (64 * 2)

            if result >= threshold:

                if boxes[i]['probabily']> boxes[j]['probabily']:

                    remove.append(j)

                else:

                    remove.append(i)

                    break

# Create figure and axes

fig,ax = plt.subplots(1)



# Display the image

ax.imshow(im)



# Create a Rectangle patch

for j,i in enumerate(boxes):

    if j not in remove:

        rect = patches.Rectangle((int(i['location'][1][0] * (1 + i['pic_reducing'])), 

                                  int(i['location'][0][0]) * (1 + i['pic_reducing']))

                                 ,int(64 * (1 + i['pic_reducing'])),

                                 int(64 * (1 + i['pic_reducing'])),

                                 linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)

# Add the patch to the Axes





plt.show()
# Set the location and name of the cfg file

cfg_file = '../input/configfiles/yolo_v3.conf'

class_name = '../input/justoneclass/classes'
from utils import *

class_names = model_v1.load_class_names(class_name)