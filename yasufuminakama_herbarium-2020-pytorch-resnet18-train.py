import os

import numpy as np 

import pandas as pd 

import json
%%time



with open('../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json', "r", encoding="ISO-8859-1") as file:

    train = json.load(file)



train_img = pd.DataFrame(train['images'])

train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')

train_df = train_img.merge(train_ann, on='id')

train_df.head()
%%time



with open('../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json', "r", encoding="ISO-8859-1") as file:

    test = json.load(file)



test_df = pd.DataFrame(test['images'])

test_df.head()
train_df['category_id'].value_counts()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

le.fit(train_df['category_id'])

train_df['category_id'] = le.transform(train_df['category_id'])
# ====================================================

# Library

# ====================================================



import sys



import gc

import os

import random

import time

from contextlib import contextmanager

from pathlib import Path



import cv2

from PIL import Image

import numpy as np

import pandas as pd

import scipy as sp



import sklearn.metrics



from functools import partial



import torch

import torch.nn as nn

from torch.optim import Adam, SGD

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, Dataset



from albumentations import Compose, Normalize, Resize

from albumentations.pytorch import ToTensorV2





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
# ====================================================

# Utils

# ====================================================



@contextmanager

def timer(name):

    t0 = time.time()

    LOGGER.info(f'[{name}] start')

    yield

    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')



    

def init_logger(log_file='train.log'):

    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler

    

    log_format = '%(asctime)s %(levelname)s %(message)s'

    

    stream_handler = StreamHandler()

    stream_handler.setLevel(DEBUG)

    stream_handler.setFormatter(Formatter(log_format))

    

    file_handler = FileHandler(log_file)

    file_handler.setFormatter(Formatter(log_format))

    

    logger = getLogger('Herbarium')

    logger.setLevel(DEBUG)

    logger.addHandler(stream_handler)

    logger.addHandler(file_handler)

    

    return logger



LOG_FILE = 'train.log'

LOGGER = init_logger(LOG_FILE)





def seed_torch(seed=777):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 777

seed_torch(SEED)
N_CLASSES = 32093





class TrainDataset(Dataset):

    def __init__(self, df, labels, transform=None):

        self.df = df

        self.labels = labels

        self.transform = transform

        

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        file_name = self.df['file_name'].values[idx]

        file_path = f'../input/herbarium-2020-fgvc7/nybg2020/train/{file_name}'

        image = cv2.imread(file_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        label = self.labels.values[idx]

        

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

        

        return image, label
HEIGHT = 128

WIDTH = 128





def get_transforms(*, data):

    

    assert data in ('train', 'valid')

    

    if data == 'train':

        return Compose([

            Resize(HEIGHT, WIDTH),

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

        ])

    

    elif data == 'valid':

        return Compose([

            Resize(HEIGHT, WIDTH),

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

        ])
from sklearn.model_selection import StratifiedKFold



DEBUG = False



if DEBUG:

    folds = train_df.sample(n=10000, random_state=0).reset_index(drop=True).copy()

else:

    folds = train_df.copy()

train_labels = folds['category_id'].values

kf = StratifiedKFold(n_splits=2)

for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):

    folds.loc[val_index, 'fold'] = int(fold)

folds['fold'] = folds['fold'].astype(int)

folds.to_csv('folds.csv', index=None)

folds.head()
FOLD = 0

trn_idx = folds[folds['fold'] != FOLD].index

val_idx = folds[folds['fold'] == FOLD].index

print(trn_idx.shape, val_idx.shape)
train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), 

                             folds.loc[trn_idx]['category_id'], 

                             transform=get_transforms(data='train'))

valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 

                             folds.loc[val_idx]['category_id'], 

                             transform=get_transforms(data='valid'))
batch_size = 512



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
import torchvision.models as models



model = models.resnet18(pretrained=True)

model.avgpool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score

from tqdm import tqdm





with timer('Train model'):

    

    n_epochs = 1

    lr = 4e-4

    

    model.to(device)

    

    optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5, verbose=True, eps=1e-6)

    

    criterion = nn.CrossEntropyLoss()

    best_score = 0.

    best_loss = np.inf

    

    for epoch in range(n_epochs):

        

        start_time = time.time()



        model.train()

        avg_loss = 0.



        optimizer.zero_grad()



        for i, (images, labels) in tqdm(enumerate(train_loader)):



            images = images.to(device)

            labels = labels.to(device)

            

            y_preds = model(images)

            loss = criterion(y_preds, labels)

            

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()



            avg_loss += loss.item() / len(train_loader)

            

        model.eval()

        avg_val_loss = 0.

        preds = np.zeros((len(valid_dataset)))



        for i, (images, labels) in enumerate(valid_loader):

            

            images = images.to(device)

            labels = labels.to(device)

            

            with torch.no_grad():

                y_preds = model(images)

            

            preds[i * batch_size: (i+1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()



            loss = criterion(y_preds, labels)

            avg_val_loss += loss.item() / len(valid_loader)

        

        scheduler.step(avg_val_loss)

            

        score = f1_score(folds.loc[val_idx]['category_id'].values, preds, average='macro')



        elapsed = time.time() - start_time



        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  F1: {score:.6f}  time: {elapsed:.0f}s')



        if score>best_score:

            best_score = score

            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.6f} Model')

            torch.save(model.state_dict(), f'fold{FOLD}_best_score.pth')



        if avg_val_loss<best_loss:

            best_loss = avg_val_loss

            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

            torch.save(model.state_dict(), f'fold{FOLD}_best_loss.pth')