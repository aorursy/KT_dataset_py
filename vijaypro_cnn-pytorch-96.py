# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required packages
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score
import seaborn as sns
# Read the csv file and print the first 10 rows
csv=pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
print(csv.head(10))
labels=csv['label']
# Plotting the number of data in each label as a countplot.
plt.figure(figsize=(10,10))
sns.countplot(labels)
# Split the label (1st Column) and Image Pixels (2nd-785th Column)
text="pixel"
images=torch.zeros((csv.shape[0],1))
for i in range(1,785):
    temp_text=text+str(i)
    temp=csv[temp_text]
    temp=torch.FloatTensor(temp).unsqueeze(1)
    images=torch.cat((images,temp),1)
images_final=torch.FloatTensor(images[:,1:]).view(-1,28,28)
fig=plt.figure(figsize=(10,10))
columns=3
rows=4
for i in range(12):
    img=images_final[i,:]
    img=img.numpy()
    img=cv2.resize(img,(224,224))
    fig.add_subplot(columns, rows, i + 1)
    plt.imshow(img)
plt.show()
class GestureDataset(Dataset):
    def __init__(self,csv,train=True):
        self.csv=pd.read_csv(csv)
        self.img_size=224
        # print(self.csv['image_names'][:5])
        self.train=train
        text="pixel"
        self.images=torch.zeros((self.csv.shape[0],1))
        for i in range(1,785):
            temp_text=text+str(i)
            temp=self.csv[temp_text]
            temp=torch.FloatTensor(temp).unsqueeze(1)
            self.images=torch.cat((self.images,temp),1)
        self.labels=self.csv['label']
        self.images=self.images[:,1:]
        self.images=self.images.view(-1,28,28)
        
    def __getitem__(self,index):
        img=self.images[index]
        img=img.numpy()
        img=cv2.resize(img,(self.img_size,self.img_size))
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.unsqueeze(0)
        tensor_image/=255.
        if self.train:
            return tensor_image,self.labels[index]
        else:
            return tensor_image
    def __len__(self):
        return self.images.shape[0]
# Using custom GestureDataset class to load train and test data respectively.
data=GestureDataset("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
data_val=GestureDataset("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")
# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=128,num_workers=4,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=64,num_workers=0,shuffle=True)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 32, 5), # 220, 220
        nn.MaxPool2d(2), # 110, 110
        nn.ReLU(),
        nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5), # 106, 106
        nn.MaxPool2d(2),  # 53,53
        nn.ReLU(),
        nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
        nn.Conv2d(64, 128, 3), # 51, 51
        nn.MaxPool2d(2), # 25, 25
        nn.ReLU(),
        nn.BatchNorm2d(128)
        )
        self.Conv4 = nn.Sequential(
        nn.Conv2d(128, 256, 3), # 23, 23
        nn.MaxPool2d(2), # 11, 11
        nn.ReLU(),            
        nn.BatchNorm2d(256)
        )
        self.Conv5 = nn.Sequential(
        nn.Conv2d(256, 512, 3), # 9, 9
        nn.MaxPool2d(2), # 4, 4
        nn.ReLU(),
        nn.BatchNorm2d(512)
        )
        
        self.Linear1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout=nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, 25)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x=self.dropout(x)
        x = self.Conv5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x
# Validating the model against the validation dataset and generate the accuracy and F1-Score.
def validate(val_loader,model):
    model.eval()
    test_labels=[0]
    test_pred=[0]
    for i, (images,labels) in enumerate(val_loader):
        outputs=model(images.to(device))
        predicted = torch.softmax(outputs,dim=1)
        _,predicted=torch.max(predicted, 1)
        test_pred.extend(list(predicted.data.cpu().numpy()))
        test_labels.extend(list(labels.data.cpu().numpy()))

    test_pred=np.array(test_pred[1:])
    test_labels=np.array(test_labels[1:])
    correct=(test_pred==test_labels).sum()
    accuracy=correct/len(test_labels)
    f1_test=f1_score(test_labels,test_pred,average='weighted')
    model.train()
    return accuracy,f1_test 
model=Classifier()
model=model.to("cuda")
model.train()
checkpoint=None
device="cuda"
learning_rate=1e-3
start_epoch=0
end_epoch=20
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose= True, min_lr=1e-6)
if checkpoint:
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    start_epoch=torch.load(checkpoint)['epoch']
for epoch in range(start_epoch,end_epoch+1):
    for i, (images,labels) in enumerate(train_loader):
        outputs=model(images.to(device))
        loss=criterion(outputs.to(device),labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.softmax(outputs,dim=1)
        _,predicted=torch.max(predicted, 1)
        f1=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')
    val_accuracy, val_f1=validate(val_loader,model)
    print("------------------------------------------------------------------------------------------------------")
    print("Epoch [{}/{}], Training F1: {:.4f}, Validation Accuracy: {:.4f}, Validation F1: {:.4f}".format(epoch,end_epoch,f1,val_accuracy,val_f1))
    scheduler.step(val_accuracy)
# Save the model for future use and optimization.
torch.save({
'epoch': epoch,
'state_dict': model.state_dict(),
'optimizer' : optimizer.state_dict()},
'checkpoint.epoch.1.{}.pth.tar'.format(epoch))
