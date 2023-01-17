import os

import numpy as np 

import pandas as pd 

import json
os.listdir('../input/imet-2020-fgvc7')
submission = pd.read_csv('../input/imet-2020-fgvc7/sample_submission.csv')
submission.head()
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

from collections import defaultdict, Counter



import cv2

from PIL import Image

import numpy as np

import pandas as pd

import scipy as sp



import sklearn.metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold



from functools import partial

from tqdm import tqdm



import torch

import torch.nn as nn

from torch.optim import Adam, SGD

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset

import torchvision.models as models



from albumentations import Compose, Normalize, Resize, RandomResizedCrop

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
N_CLASSES = 3474





class TrainDataset(Dataset):

    def __init__(self, df, labels, transform=None):

        self.df = df

        self.labels = labels

        self.transform = transform

        

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        file_name = self.df['id'].values[idx]

        file_path = f'../input/imet-2020-fgvc7/train/{file_name}.png'

        image = cv2.imread(file_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

            

        label = self.labels.values[idx]

        target = torch.zeros(N_CLASSES)

        for cls in label.split():

            target[int(cls)] = 1

        

        return image, target

    



class TestDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df

        self.transform = transform

        

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        file_name = self.df['id'].values[idx]

        file_path = f'../input/imet-2020-fgvc7/test/{file_name}.png'

        image = cv2.imread(file_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

        

        return image
HEIGHT = 128

WIDTH = 128





def get_transforms(*, data):

    

    assert data in ('train', 'valid')

    

    if data == 'train':

        return Compose([

            #Resize(HEIGHT, WIDTH),

            RandomResizedCrop(HEIGHT, WIDTH),

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
batch_size = 128



test_dataset = TestDataset(submission, transform=get_transforms(data='valid'))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = models.resnet18(pretrained=False)

model.avgpool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, N_CLASSES)



weights_path = '../input/imet-2020-pytorch-resnet18-starter/fold0_best_score.pth'

model.load_state_dict(torch.load(weights_path))
with open('../input/imet-2020-pytorch-resnet18-starter/train.log') as f:

    s = f.read()

    print(s)
with timer('inference'):

    

    model.to(device) 

    

    preds = []

    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))



    for i, images in tk0:

            

        images = images.to(device)

            

        with torch.no_grad():

            y_preds = model(images)

            

        preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
threshold = 0.09

predictions = np.concatenate(preds) > threshold



for i, row in enumerate(predictions):

    ids = np.nonzero(row)[0]

    submission.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])

    

submission.to_csv('submission.csv', index=False)

submission.head()