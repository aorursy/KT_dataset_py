# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
digit_recon_tran_csv = pd.read_csv('/kaggle/input/digit-recognizer/train.csv',dtype = np.float32)

digit_recon_test_csv = pd.read_csv('/kaggle/input/digit-recognizer/test.csv',dtype = np.float32)
print('tran dataset size: ',digit_recon_tran_csv.size,'\n')

print('test dataset size: ',digit_recon_test_csv.size,'\n')
#print(digit_recon_tran_csv.head(1))

#print(digit_recon_tran_csv.head(1).label)

tran_label = digit_recon_tran_csv.label.values

tran_image = digit_recon_tran_csv.loc[:,digit_recon_tran_csv.columns != "label"].values/255 # normalization

test_image = digit_recon_test_csv.values/255
print('train label size: ',tran_label.shape)

print('train image size: ',tran_image.shape)

print('test  image size: ',test_image.shape)
from sklearn.model_selection import train_test_split

train_image, valid_image, train_label, valid_label = train_test_split(tran_image,

                                                                      tran_label,

                                                                      test_size = 0.2,

                                                                      random_state = 42) #
print('train size: ',train_image.shape)

print('valid size: ',valid_image.shape)
import torch 

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms



print(torch.__version__)





class MNIST_data(Dataset):

    """MNIST dtaa set"""

    

    def __init__(self, 

                 data, 

                 transform = transforms.Compose([transforms.ToPILImage(),

                                                 transforms.RandomAffine(30,(0.1,0.1)),

                                                 transforms.ToTensor()

                                                ])

                ):

        

        if len(data) == 1:

            # test data

            self.X = data[0].reshape(-1,28,28)

            self.y = None

        else:

            # training data

            self.X = data[0].reshape(-1,28,28)

            self.y = data[1].astype(np.long)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
# test mnist dataset

import matplotlib.pyplot as plt

test_mnist_data = MNIST_data((train_image,train_label))

test_mnist_loader = torch.utils.data.DataLoader(dataset=test_mnist_data,

                                           batch_size=1, shuffle=True)

for batch_idx, (images, labels) in enumerate(test_mnist_loader):

                

        plt.imshow(images.view(28,28).numpy())

        plt.axis("off")

        plt.title(str(labels.numpy()))

        plt.show()

        

        break
# visual

import matplotlib.pyplot as plt

plt.imshow(test_image[10].reshape(28,28))

plt.axis("off")

plt.show()
batch_size = 64 # 2^5=64



train_dataset = MNIST_data((train_image,train_label))

train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size, shuffle=True)



valid_dataset = MNIST_data((valid_image,valid_label))

valid_loader  = torch.utils.data.DataLoader(dataset=valid_dataset,

                                           batch_size=batch_size, shuffle=False)
class YANNet(nn.Module):

    def __init__(self):

        super(YANNet,self).__init__()

        

        self.conv = nn.Sequential( 

            # size: 28*28

            nn.Conv2d(1,8,3,1,1), # in_channels out_channels kernel_size stride padding

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.Conv2d(8,16,3,1,1), 

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # size: 14*14

            nn.Conv2d(16,16,3,1,1), 

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Conv2d(16,8,3,1,1), 

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        

        self.fc = nn.Sequential(

            # size: 7*7

            nn.Linear(8*7*7,256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256,256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256,10)

        )



    

    def forward(self, img):

        x = self.conv(img)

        o = self.fc(x.view(x.shape[0],-1))

        return o
model = YANNet()

error = nn.CrossEntropyLoss()



if torch.cuda.is_available():

    model = model.cuda()

    error = error.cuda()

    

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_epoc = 120

from torch.autograd import Variable



for epoch in range(num_epoc):

    epoc_train_loss = 0.0

    epoc_train_corr = 0.0

    epoc_valid_corr = 0.0

    print('Epoch:{}/{}'.format(epoch,num_epoc))

    

    model.train()

    scheduler.step()

    

    for batch_idx, (images, labels) in enumerate(train_loader):

        

        if torch.cuda.is_available():

            images = images.cuda()

            labels = labels.cuda()

        images = Variable(images)

        labels = Variable(labels)

        

        outputs = model(images)

               

        optimizer.zero_grad()

        loss = error(outputs,labels)

        loss.backward()

        optimizer.step()

        

        epoc_train_loss += loss.data

        outputs = torch.max(outputs.data,1)[1]

        epoc_train_corr += torch.sum(outputs==labels.data)

    

    with torch.no_grad():

        model.eval()

        for batch_idx, (images, labels) in enumerate(valid_loader):



            if torch.cuda.is_available():

                images = images.cuda()

                labels = labels.cuda()

            images = Variable(images)

            labels = Variable(labels)





            outputs = model(images)

            outputs = torch.max(outputs.data,1)[1]



            epoc_valid_corr += torch.sum(outputs==labels.data)

    

    

    print("loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(epoc_train_loss/len(train_dataset),100*epoc_train_corr/len(train_dataset),100*epoc_valid_corr/len(valid_dataset)))
model = model.cpu()

model.eval()
plt.imshow(test_image[100].reshape(28,28))

plt.axis("off")

plt.show()



one_test = test_image[100]

one_test = torch.from_numpy(one_test).view(1,1,28,28)

one_output = model(one_test)

print(torch.max(one_output.data,1)[1].numpy())

digit_recon_submission_csv = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv',dtype = np.float32)

print(digit_recon_submission_csv.head(10))
print(test_image.shape)
test_results = np.zeros((test_image.shape[0],2),dtype='int32')

print(test_results.shape)
for i in range(test_image.shape[0]): 

    one_image = torch.from_numpy(test_image[i]).view(1,1,28,28)

    one_output = model(one_image)

    test_results[i,0] = i+1

    test_results[i,1] = torch.max(one_output.data,1)[1].numpy()

    
print(test_results.shape)
Data = {'ImageId': test_results[:, 0], 'Label': test_results[:, 1]}

DataFrame = pd.DataFrame(Data)

DataFrame.to_csv('submission.csv', index=False, sep=',')