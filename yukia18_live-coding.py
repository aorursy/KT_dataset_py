import os

import random



import cv2

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



from tqdm import tqdm



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms





%matplotlib inline
train_df = pd.read_csv('../input/ailab-ml-training-1/train.csv')

test_df = pd.read_csv('../input/ailab-ml-training-1/sample_submission.csv')
train_df.head()
test_df.head()
train_df.shape, test_df.shape
img = cv2.imread('../input/ailab-ml-training-1/train_images/train_images/train_0.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap='gray')

plt.show()
img.shape
class KmnistDataset(Dataset):

    def __init__(self, fnames, labels, root_dir, transform=None):

        self.fnames = fnames

        self.labels = labels

        self.root_dir = root_dir

        self.transform = transform

        

        self.images = []

        for fn in tqdm(fnames):

            self.images.append(self.__load_image(fn))

    

    def __len__(self):

        return len(self.fnames)

        

    def __load_image(self, fname):

        path = os.path.join(self.root_dir, fname)

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    

    def __getitem__(self, idx):

        img = self.images[idx]

        label = self.labels[idx]

        

        if self.transform is not None:

            img = self.transform(img)

        

        return img, label
class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_block = nn.Sequential(

            # (B, 1, 28, 28) --> (B, 32, 28, 28) --> (B, 32, 14, 14)

            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d((2, 2)),

            # (B, 32, 14, 14) --> (B, 64, 14, 14) --> (B, 64, 7, 7)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d((2, 2)),

            # (B, 64, 7, 7) --> (B, 128, 7, 7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.ReLU(),

        )

        # (B, 128, 7, 7) --> (B, 128 * 7 * 7) --> (B, 10)

        self.fc = nn.Linear(128 * 7 * 7, 10)

    

    def forward(self, x):

        x = self.conv_block(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
train_df, valid_df = train_test_split(

    train_df, test_size=0.2, random_state=42, shuffle=True

)
train_df.shape, valid_df.shape
train_df = train_df.reset_index(drop=True)

valid_df = valid_df.reset_index(drop=True)
train_df.head()
transform = transforms.Compose([

    transforms.ToTensor(),

])



train_dataset = KmnistDataset(

    train_df['fname'].tolist(),

    train_df['label'].tolist(),

    root_dir='../input/ailab-ml-training-1/train_images/train_images/',

    transform=transform

)

valid_dataset = KmnistDataset(

    valid_df['fname'].tolist(),

    valid_df['label'].tolist(),

    root_dir='../input/ailab-ml-training-1/train_images/train_images/',

    transform=transform

)



train_dataloader = DataLoader(

    train_dataset,

    batch_size=64,

    shuffle=True,

)

valid_dataloader = DataLoader(

    valid_dataset,

    batch_size=64,

    shuffle=False,

)
model = ConvNet()

optimizer = Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
model = model.to(device='cuda')
for epoch in range(5):

    # train

    model.train()

    train_loss_list = []

    for x, y in train_dataloader:

        x = x.to(device='cuda')

        y = y.to(device='cuda')

        

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        

        train_loss_list.append(loss.item())

        

    # valid

    model.eval()

    valid_loss_list = []

    for x, y in valid_dataloader:

        x = x.to(device='cuda')

        y = y.to(device='cuda')

        

        with torch.no_grad():

            pred = model(x)

            loss = criterion(pred, y)

        

        valid_loss_list.append(loss.item())



    # logging

    print('epoch: {} - loss - {:.5f} - val_loss - {:.5f}'.format(

        epoch,

        np.mean(train_loss_list),

        np.mean(valid_loss_list),

    ))
test_df.head()
test_df.shape
transform = transforms.Compose([

    transforms.ToTensor(),

])



test_dataset = KmnistDataset(

    test_df['fname'].tolist(),

    test_df['label'].tolist(),

    root_dir='../input/ailab-ml-training-1/test_images/test_images/',

    transform=transform

)



test_dataloader = DataLoader(

    test_dataset,

    batch_size=64,

    shuffle=False,

)
model.eval()

predictions = []



for x, _ in tqdm(test_dataloader):

    x = x.to(device='cuda')

        

    with torch.no_grad():

        pred = model(x)

        # (B, 10)

        pred = torch.argmax(pred, dim=1).cpu().numpy()

        predictions += pred.tolist()
test_df['label'] = predictions
test_df.head()
test_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink('submission.csv')