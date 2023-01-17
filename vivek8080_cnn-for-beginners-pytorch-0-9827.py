from os import path

import torch

import torch, torchvision

import seaborn as sns

from torch.autograd import Variable

from tqdm import tqdm

import matplotlib.pyplot as plt

import pandas as pd

from torch.utils.data import DataLoader, Dataset, sampler

from torchvision import  transforms

import numpy as np

from torch import nn,optim
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')

submission=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
#checking for null value

print('Null in train_data {}'.format(train.isna().any().sum()))

print('Null in test_data {}'.format(test.isna().any().sum()))
train.head(2)
#Looking for first number

plt.imshow(train.loc[0,train.columns!='label'].values.reshape(28,28)) #for image

plt.title(train.loc[0,train.columns=='label'].values[0])              #For target

plt.show()
train['label'].value_counts()   #unique labels and there count
sns.countplot(x='label',data=train) #count of target
class DigitDataset(Dataset):

    def __init__(self,df,trans):

        self.df=df

        self.transform=trans

        self.fnames = self.df.index.tolist()

    def __getitem__(self,idx):

        df=self.df

        img=df.loc[idx,df.columns!='label'].values.astype('float32')/255

        target=train.loc[idx,train.columns=='label'].values[0]

        img=img.reshape(28,28)

        img=self.transform(img)

        target=torch.tensor(target).type(torch.LongTensor)

       # print(target.dtype)

        return img,target

    def __len__(self):

        return len(self.fnames)

data_tranf=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
data=DigitDataset(train,data_tranf)

data_loader=DataLoader(data,batch_size=10,shuffle=False)
#lets see our Dataloader is working or not

img,target=data.__getitem__(0)

img.shape,target.shape
class CnnNetwork(nn.Module):

    def __init__(self):

        super(CnnNetwork,self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)

        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1=nn.Linear(in_features=12*20*20,out_features=120)

        self.fc2=nn.Linear(in_features=120,out_features=60)

        self.out=nn.Linear(in_features=60,out_features=10)

        self.relu=nn.ReLU()

    def forward(self,x):

        n=x.size(0)

        x=self.relu(self.conv1(x))

        x=self.relu(self.conv2(x))

        x=x.view(n,-1)

        x=self.fc1(x)

        x=self.fc2(x)

        x=self.out(x)

        return x
model = CnnNetwork()

cec_loss = nn.CrossEntropyLoss()

params = model.parameters()

optimizer = optim.Adam(params=params,lr=0.001)
n_epochs=20

n_iterations=2



# vis=Visdom()

# vis_window=vis.line(np.array([0]),np.array([0]))



for e in range(n_epochs):

    for i,(images,labels) in enumerate(data_loader):

        images = Variable(images)

        labels = Variable(labels)

        output = model(images)

        

        model.zero_grad()

        loss = cec_loss(output,labels)

 #       print(loss)

        loss.backward()

        

        optimizer.step()

        

        n_iterations+=1

    print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))
test_d=DigitDataset(test,data_tranf)

test_loader=DataLoader(test_d,batch_size=test.shape[0],shuffle=False)

for i,(test_img,lab) in enumerate(test_loader):

    print(i)

    break

output=model(test_img)

predicted = torch.max(output,1)[1]

submission['Label']=predicted
submission.to_csv('submission1.csv',index=False)