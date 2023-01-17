import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm, notebook

# train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

# data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

# data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

# data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

# data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

data0 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_0.feather')

data1 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_1.feather')

data2 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_2.feather')

data3 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_3.feather')
import cv2

def Resize(df,size=64):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
# data0 = Resize(data0)
data_full = pd.concat([data0, data1, data2, data3],ignore_index=True)

# data_full = data0
class GraphemeDataset(Dataset):

    def __init__(self,df,label,_type='train'):

        self.df = df

        self.label = label

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        label1 = self.label.vowel_diacritic.values[idx]

        label2 = self.label.grapheme_root.values[idx]

        label3 = self.label.consonant_diacritic.values[idx]

        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(np.float)

        return image,label1,label2,label3
class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):

        super(ResidualBlock,self).__init__()

        self.cnn1 =nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

        )

        self.cnn2 = nn.Sequential(

            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),

            nn.BatchNorm2d(out_channels)

        )

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Sequential()

            

    def forward(self,x):

        residual = x

        x = self.cnn1(x)

        x = self.cnn2(x)

        x += self.shortcut(residual)

        x = nn.ReLU(True)(x)

        return x
class ResNet18(nn.Module):

    def __init__(self):

        super(ResNet18,self).__init__()

        

        self.block1 = nn.Sequential(

            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(True)

        )

        

        self.block2 = nn.Sequential(

            nn.MaxPool2d(1,1),

            ResidualBlock(64,64),

            ResidualBlock(64,64,2)

        )

        

        self.block3 = nn.Sequential(

            ResidualBlock(64,128),

            ResidualBlock(128,128,2)

        )

        

        self.block4 = nn.Sequential(

            ResidualBlock(128,256),

            ResidualBlock(256,256,2)

        )

        self.block5 = nn.Sequential(

            ResidualBlock(256,512),

            ResidualBlock(512,512,2)

        )

        

        self.avgpool = nn.AvgPool2d(2)

        # vowel_diacritic

        self.fc1 = nn.Linear(512,11)

        # grapheme_root

        self.fc2 = nn.Linear(512,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(512,7)

        

    def forward(self,x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.avgpool(x)

        x = x.view(x.size(0),-1)

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        return x1,x2,x3
# !pip install torchsummary

# from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet18().to(device)

# summary(model, (1, 64, 64))
model = ResNet18().to(device)

optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)

criterion = nn.CrossEntropyLoss()

batch_size=32
epochs = 50

model.train()

losses = []

accs = []

for epoch in range(epochs):

    reduced_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

    reduced_train = train.loc[train.image_id.isin(reduced_index)]

    reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

    train_image = GraphemeDataset(reduced_data,reduced_train)

    train_loader = torch.utils.data.DataLoader(train_image,batch_size=batch_size,shuffle=True)

    

    print('epochs {}/{} '.format(epoch+1,epochs))

    running_loss = 0.0

    running_acc = 0.0

    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)

        

        optimizer.zero_grad()

        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

        loss1 = criterion(outputs1,labels1)

        loss2 = criterion(outputs2,labels2)

        loss3 = criterion(outputs3,labels3)

        running_loss += loss1+loss2+loss3

        running_acc += (outputs1.argmax(1)==labels1).float().mean()

        running_acc += (outputs2.argmax(1)==labels2).float().mean()

        running_acc += (outputs3.argmax(1)==labels3).float().mean()

        (loss1+loss2+loss3).backward()

        optimizer.step()

    #scheduler.step()

    losses.append(running_loss/len(train_loader))

    accs.append(running_acc/(len(train_loader)*3))

    print('acc : {:.2f}%'.format(running_acc/(len(train_loader)*3)))

    print('loss : {:.4f}'.format(running_loss/len(train_loader)))

torch.save(model.state_dict(), 'saved_weights.pth')
test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

test_data0 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/test_data_0.feather')

test_data1 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/test_data_1.feather')

test_data2 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/test_data_2.feather')

test_data3 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/test_data_3.feather')
test_data_full = pd.concat([test_data0, test_data1, test_data2, test_data3],ignore_index=True)
accs = []

# reduced_index =test.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

# reduced_test = test.loc[test.image_id.isin(reduced_index)]

# reduced_data = test_data_full.loc[test_data_full.image_id.isin(reduced_index)]

# test_image = GraphemeDataset(test_data_full,test)

# test_loader = torch.utils.data.DataLoader(test_image,batch_size=batch_size,shuffle=True)

test_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(1)).image_id.values

reduced_test = train.loc[train.image_id.isin(test_index)]

test_data = data_full.loc[data_full.image_id.isin(test_index)]



test_image = GraphemeDataset(test_data,reduced_test)

test_loader = torch.utils.data.DataLoader(test_image,batch_size=batch_size,shuffle=True)



with torch.no_grad():

    running_acc = 0.0

    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(test_loader),total=len(test_loader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)



        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

        running_acc += (outputs1.argmax(1)==labels1).float().mean()

        running_acc += (outputs2.argmax(1)==labels2).float().mean()

        running_acc += (outputs3.argmax(1)==labels3).float().mean()



    accs.append(running_acc/(len(train_loader)*3))

    print('acc : {:.2f}%'.format(running_acc/(len(test_loader)*3)))



# 