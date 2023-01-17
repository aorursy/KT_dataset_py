import os

import random



import numpy as np

import pandas as pd

import cv2



import matplotlib.pyplot as plt

import seaborn as sns



from fastprogress import master_bar, progress_bar



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms





%matplotlib inline
def seed_everything(seed=42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
DIRPATH = '../input/ailab-ml-training-0/'

TRAIN_IMAGE_DIR = 'train_images/train_images/'

TEST_IMAGE_DIR = 'test_images/test_images/'



ID = 'fname'

TARGET = 'label'



VALID_SIZE = 0.2

EPOCHS = 5

BATCH_SIZE = 64

LR = 1e-3



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



SEED = 42

seed_everything(SEED)
os.listdir(DIRPATH)
train_df = pd.read_csv(os.path.join(DIRPATH, 'train.csv'))
train_df.head()
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4 * len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = train_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(DIRPATH, TRAIN_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - label: {label}')



plt.show()
class TrainDataset(Dataset):

    def __init__(self, fname_list, label_list, image_dir, transform=None):

        super().__init__()

        self.fname_list = fname_list

        self.label_list = label_list

        self.image_dir = image_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.fname_list)

    

    def __getitem__(self, idx):

        fname = self.fname_list[idx]

        label = self.label_list[idx]

        

        image = cv2.imread(os.path.join(self.image_dir, fname))

        if self.transform is not None:

            image = self.transform(image)

        

        return image, label
class SimpleClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_module = nn.Sequential(

            # (N, 3, 28, 28) --> (N, 32, 14, 14)

            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True),

            nn.ReLU(True),

            nn.MaxPool2d(2),

            # (N, 32, 14, 14) --> (N, 64, 7, 7)

            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True),

            nn.ReLU(True),

            nn.MaxPool2d(2),

            # (N, 64, 7, 7) --> (N, 128, 7, 7)

            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True),

            nn.ReLU(True),

        )

        self.dense_module = nn.Sequential(

            nn.Linear(128*7*7, 10, bias=True)

        )

    

    def forward(self, x):

        x = self.conv_module(x)

        x = x.view(x.size(0), -1)

        x = self.dense_module(x)



        return x
fname_list = train_df[ID].to_list()

label_list = train_df[TARGET].to_list()



train_fname_list, valid_fname_list, train_label_list, valid_label_list = train_test_split(

    fname_list, label_list, test_size=VALID_SIZE, random_state=SEED, shuffle=True

)
len(fname_list), len(train_fname_list), len(valid_fname_list)
image_dir = os.path.join(DIRPATH, TRAIN_IMAGE_DIR)



transform = transforms.Compose([

    transforms.ToTensor()

])



train_dataset = TrainDataset(train_fname_list, train_label_list, image_dir, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



valid_dataset = TrainDataset(valid_fname_list, valid_label_list, image_dir, transform=transform)

valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = SimpleClassifier().to(DEVICE)

optim = Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()
mb = master_bar(range(EPOCHS))



for epoch in mb:

    

    # training

    

    model.train()

    train_loss_list = []

    train_accuracy_list = []

    

    for batch_image, batch_label in progress_bar(train_dataloader, parent=mb):

        batch_image = batch_image.to(dtype=torch.float32, device=DEVICE)

        batch_label = batch_label.to(dtype=torch.long, device=DEVICE)

        

        optim.zero_grad()

        batch_pred = model(batch_image)

        loss = criterion(batch_pred, batch_label)

        loss.backward()

        optim.step()

        

        train_loss_list.append(loss.item())

        accuracy = accuracy_score(torch.argmax(batch_pred, axis=1).cpu().numpy(), batch_label.cpu().numpy())

        train_accuracy_list.append(accuracy)

    

    # validation

    

    model.eval()

    valid_loss_list = []

    valid_accuracy_list = []

    

    for batch_image, batch_label in valid_dataloader:

        batch_image = batch_image.to(dtype=torch.float32, device=DEVICE)

        batch_label = batch_label.to(dtype=torch.long, device=DEVICE)

        

        with torch.no_grad():

            batch_pred = model(batch_image)

            loss = criterion(batch_pred, batch_label)

        

        valid_loss_list.append(loss.item())

        accuracy = accuracy_score(torch.argmax(batch_pred, axis=1).cpu().numpy(), batch_label.cpu().numpy())

        valid_accuracy_list.append(accuracy)

    

    # verbose

    

    mb.write('epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(

        epoch,

        EPOCHS, 

        np.mean(train_loss_list),

        np.mean(train_accuracy_list),

        np.mean(valid_loss_list),

        np.mean(valid_accuracy_list)

    ))
submission_df = pd.read_csv(os.path.join(DIRPATH, 'sample_submission.csv'))
submission_df.head()
fname_list = submission_df[ID].to_list()

label_list = submission_df[TARGET].to_list()



image_dir = os.path.join(DIRPATH, TEST_IMAGE_DIR)



transform = transforms.Compose([transforms.ToTensor()])



test_dataset = TrainDataset(fname_list, label_list, image_dir, transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()

predictions = []



for batch_image, _ in progress_bar(test_dataloader):

    batch_image = batch_image.to(dtype=torch.float32, device=DEVICE)

    

    with torch.no_grad():

        batch_pred = model(batch_image)

        batch_pred = torch.argmax(batch_pred, axis=1).cpu().numpy()

        

    predictions.append(batch_pred[0])
submission_df[TARGET] = predictions
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4 * len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = submission_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(DIRPATH, TEST_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - label: {label}')



plt.show()
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink('submission.csv')