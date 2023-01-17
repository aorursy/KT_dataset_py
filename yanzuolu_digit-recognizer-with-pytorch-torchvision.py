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
import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torchvision

from torchvision import transforms

from torchvision import models

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

from PIL import Image

from time import sleep

import matplotlib.pyplot as plt

import seaborn as sns

import ssl

%matplotlib inline

ssl._create_default_https_context = ssl._create_unverified_context
train_pth = "/kaggle/input/digit-recognizer/train.csv"

test_pth = "/kaggle/input/digit-recognizer/test.csv"

train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)

print("train: {}".format(len(train)))

print("test: {}".format(len(test)))
class MNIST(Dataset):

    def __init__(self, df, phase, transform=None):

        self.phase = phase

        self.df = df

        self.transform = transform

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        if self.phase in ['train', 'val']:

            image = self.df.iloc[idx,1:].values.reshape((28,28)).astype(np.uint8)

#             X = Image.fromarray(np.concatenate((image,image,image),axis=2))

            X = Image.fromarray(image)

            y = np.array(self.df.iloc[idx,0])

            return self.transform(X), torch.from_numpy(y)

        else:

            image = self.df.iloc[idx].values.reshape((28,28)).astype(np.uint8)

#             X = Image.fromarray(np.concatenate((image,image,image),axis=2))

            X = Image.fromarray(image)

            return self.transform(X)
transform = transforms.Compose([

    transforms.RandomRotation(10),

    transforms.ToTensor()

])
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1,16,5,1,2),

            nn.ReLU(inplace=True),

            nn.Conv2d(16,64,5,1,2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64,128,5,1,2),

            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,5,1,2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)

        )

        self.classifier = nn.Sequential(

            nn.Linear(256 * 7 * 7, 32 * 7 * 7, bias=True),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5, inplace=False),

            nn.Linear(32 * 7 * 7, 4 * 7 * 7, bias=True),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5, inplace=False),

            nn.Linear(4 * 7 * 7, 10, bias=True)

        )

        # (input_size + padding*2 - kernel_size) / stride + 1 = output_size

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        output = self.classifier(x)

        return output

    
EPOCH = 30

BATCH_SIZE = 64

LR = 0.001
model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,verbose=True)
train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)



train, val = train_test_split(train, test_size = 0.1)



train_data = MNIST(train, 'train', transform)

test_data = MNIST(val, 'val', transform)



train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
valid_loss_min = np.Inf

train_losses, valid_losses = [], []

history_accuracy = []



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
for epoch in range(1, EPOCH+1):

    running_loss = 0

    model.train()

    try:

        with tqdm(train_loader, desc='Epoch [{}/{}]'.format(epoch, EPOCH)) as pbar1:

            for x, y in pbar1:

                x, y = x.to(device), y.to(device)



                optimizer.zero_grad()

                output = model(x)

                loss = criterion(output, y)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                pbar1.set_postfix(loss=loss.item(), refresh=True)

            train_losses.append(running_loss / len(train_loader))

            lr_scheduler.step(train_losses[-1])

            sleep(0.5)

    except KeyboardInterrupt:

        pbar1.close()

        raise

    pbar1.close()

        

    with torch.no_grad():

        try:

            with tqdm(test_loader, desc='Testing') as pbar2:

                model.eval()

                accuracy = 0

                valid_loss = 0

                for x, y in pbar2:

                    x, y = x.to(device), y.to(device)

                    test_output = model(x)

                    batch_loss = criterion(test_output, y)

                    valid_loss += batch_loss

                    pred_y = torch.max(test_output, 1)[1].data.squeeze()

                    batch_correct = pred_y.eq(y.data.view_as(pred_y)).cpu().sum()



                    batch_acc = batch_correct.item() / y.size(0)

                    pbar2.set_postfix(loss=batch_loss.item(), acc=batch_acc, refresh=True)

                    accuracy += batch_acc

                valid_losses.append(valid_loss / len(test_loader))

                history_accuracy.append(accuracy / len(test_loader))



                if valid_losses[-1] < valid_loss_min:

                    torch.save(model.state_dict(), "model.pth")

                    valid_loss_min = valid_losses[-1]

        except KeyboardInterrupt:

            pbar2.close()

            raise

        pbar2.close()

    

    print("\tEpoch[{}/{}]\ttrain_loss:{:.6f}\tvalid_loss:{:.6f}\tvalid_acc:{:.4f}".format

          (epoch,EPOCH,train_losses[-1],valid_losses[-1],history_accuracy[-1]))

    sleep(0.5)
plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(history_accuracy, label='Validation Accuracy')

plt.legend(frameon=False)
model = CNN()

model.load_state_dict(torch.load('model.pth'))

model.eval()
dataset = MNIST(test, 'test', transform)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

model.to(device)

pred_y = torch.LongTensor()



try:

    with tqdm(dataloader) as pbar3:

        for i, data in enumerate(pbar3):

            data = data.to(device)

            output = model(data)

            pred_y = torch.cat((pred_y, torch.max(output, axis=1)[1].cpu().data),dim=0)

except KeyboardInterrupt:

    pbar3.close()

    raise

pbar3.close()
submission = pd.DataFrame(np.c_[np.arange(1, len(dataset)+1)[:,None], pred_y], 

                      columns=['ImageId', 'Label'])
submission.head()
submission.to_csv('submission.csv', index=False)