import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

from torchvision import transforms,models

import os

os.listdir("../input/digit-recognizer")
train=pd.read_csv('../input/digit-recognizer/train.csv',dtype=np.float32)

final_test=pd.read_csv('../input/digit-recognizer/test.csv',dtype=np.float32)

sample_sub=pd.read_csv("../input/digit-recognizer/sample_submission.csv",

                       dtype=np.float32)

                     
train.head()
final_test.head()

sample_sub.shape
# Seperate the features and lables

# lables

targets_np=train.label.values

# scaling/255

features_np=train.loc[:,train.columns!='label'].values



# split into training and validation set

# 0.8 for training,0.2 for testing

features_train, features_test, target_train, target_test = train_test_split(

    features_np, targets_np, test_size=0.2, random_state=42)
# convert numpy to tensor

featuresTrain=torch.from_numpy(features_train)

targetsTrain=torch.from_numpy(target_train).type(torch.LongTensor)



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(target_test).type(torch.LongTensor)
# numpy

features_train=features_train.reshape(-1,1,28,28)

features_test=features_test.reshape(-1,1,28,28)
# torch

featuresTrain=featuresTrain.view(-1,1,28,28)

featuresTest=featuresTest.view(-1,1,28,28)
from torch.utils.data import Dataset,DataLoader,TensorDataset

# data augmentation

train_transform=transforms.Compose([

    transforms.ToPILImage(mode='L'),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    ]

)



test_transform = transforms.Compose([

    transforms.ToPILImage(mode='L'),                                    

    transforms.ToTensor(),

])

    

class ImgDataset(Dataset):

    def __init__(self, x, y=None, transform=None):

        self.x = x

        # label is required to be a LongTensor

        self.y = y

        if y is not None:

            self.y = torch.LongTensor(y)

        self.transform = transform

    def __len__(self):

        return len(self.x)

    # index

    def __getitem__(self, index):

        X = self.x[index]

        if self.transform is not None:

            X = self.transform(X)

        if self.y is not None:

            Y = self.y[index]

            return X, Y

        else:

            return X
# set batch size

batch_size=256



# pytorch train and test sets

train=torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test=torch.utils.data.TensorDataset(featuresTest,targetsTest)

# using numpy

#train=ImgDataset(featuresTrain,targetsTrain,train_transform)

#test=ImgDataset(featuresTest,targetsTest,test_transform)

# data loader

train_loader=torch.utils.data.DataLoader(train,

                                        batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(test,

                                       batch_size=batch_size,shuffle=False)

featuresTrain.dtype
def visualize_image(data, index, pred=False, val=0):

    '''This funtion can be used to visualize the images'''

    plt.imshow(data[index].reshape(28,28))

    plt.axis("off")

    plt.title("Handwritten Digit Image")

    plt.show()

visualize_image(features_np, 13)
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        #nn.conv2d(in_channels,out_channels,kernel_size,stride,padding)

        # 第一层卷积层(1,28,28)->(32,28,28)

        # input channel 1,output channel,kernel size 3 

        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,

                            stride=1,padding=1)

        self.batchnorm1=nn.BatchNorm2d(32)

        # 第二层卷积层(32,28,28)->(32,14,14)->(64,14,14)

        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,

                           stride=1,padding=1)

        self.batchnorm2=nn.BatchNorm2d(64)

        # 第三层卷积层(64,7,7)->(128,6,6)

        # change kernel size(3,3)

        self.conv3=nn.Conv2d(64,128,2)

        self.batchnorm3=nn.BatchNorm2d(128)

        # maxpooling(128,6,6)->(128,3,3)

        # 全连接层四层

        self.fc1=nn.Linear(128*3*3,60)

        self.fc2=nn.Linear(60,10)

        # maxpooling

        self.pool=nn.MaxPool2d(2,2)

        #dropout

        self.dropout=nn.Dropout(p=0.2)

        # output layer

        self.log_softmax=F.log_softmax

    def forward(self,x):

        # 第一层卷积

        x=self.pool(F.relu(self.batchnorm1(self.dropout(self.conv1(x)))))

        # 第二层卷积

        x=self.pool(F.relu(self.batchnorm2(self.dropout(self.conv2(x)))))

        # 第三层卷积

        x=self.pool(F.relu(self.batchnorm3(self.dropout(self.conv3(x)))))

        # 展开

        x=x.view(-1,128*3*3)

        x=self.dropout(F.relu(self.fc1(x)))

        x=self.fc2(x)

        x=self.log_softmax(x,dim=1)

        return x





device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=Net()

model=model.to(device)









from torch import optim



criterion=nn.NLLLoss()

optimizer=optim.Adam(model.parameters(),lr=0.0005)

step=0

train_losses, test_losses = [], []

epochs = 35

for e in range(epochs):

    running_loss=0.0

    for images,labels in train_loader:

        step+=1

        optimizer.zero_grad()

        log_ps=model(images.to(device))

        loss=criterion(log_ps,labels.to(device))

        # backprop

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        if step%50==0:

            test_loss=0

            accuracy=0

            with torch.no_grad():

                model.eval()

                for images,labels in test_loader:

                    images,labels=images.to(device),labels.to(device)

                    log_ps=model(images)

                    test_loss+=criterion(log_ps,labels)

                    ps = torch.exp(log_ps).to(device)

                    # Get our top predictions

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                

                model.train()

                train_losses.append(running_loss/len(train_loader))

                test_losses.append(test_loss/len(test_loader))



                print("Epoch: {}/{}.. ".format(e+1, epochs),

                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),

                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),

                  "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))



        

        

            

         

        



        

        

        
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
final_test_np = final_test.values/255

test_tn = torch.from_numpy(final_test_np)

test_tn=test_tn.view(-1,1,28,28)

fake_labels = np.zeros(final_test_np.shape)

fake_labels = torch.from_numpy(fake_labels)

submission_tn_data = torch.utils.data.TensorDataset(test_tn, fake_labels)

submission_loader = torch.utils.data.DataLoader(submission_tn_data, batch_size = batch_size, shuffle = False)
submission = [['ImageId', 'Label']]



# Turn off gradients for validation

with torch.no_grad():

    model.eval()

    image_id = 1

    for images, _ in submission_loader:

        images=images.to(device)

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1
submission_df = pd.DataFrame(submission)

submission_df.columns = submission_df.iloc[0]

submission_df = submission_df.drop(0, axis=0)

submission_df.to_csv("submission.csv", index=False)