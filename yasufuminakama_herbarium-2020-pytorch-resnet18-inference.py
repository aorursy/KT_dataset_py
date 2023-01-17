import os

import numpy as np 

import pandas as pd 

import json
os.listdir('../input/herbarium-2020-fgvc7')
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
sample_submission = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')

sample_submission.head()
train_df['category_id'].value_counts()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

le.fit(train_df['category_id'])

train_df['category_id_le'] = le.transform(train_df['category_id'])

class_map = dict(sorted(train_df[['category_id_le', 'category_id']].values.tolist()))
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





class TestDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df

        self.transform = transform

        

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        file_name = self.df['file_name'].values[idx]

        file_path = f'../input/herbarium-2020-fgvc7/nybg2020/test/{file_name}'

        image = cv2.imread(file_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

        

        return image
HEIGHT = 128

WIDTH = 128





def get_transforms():

    

    return Compose([

            Resize(HEIGHT, WIDTH),

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

        ])

batch_size = 512



test_dataset = TestDataset(test_df, transform=get_transforms())

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
import torchvision.models as models



model = models.resnet18(pretrained=False)

model.avgpool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, N_CLASSES)



weights_path = '../input/herbarium-2020-pytorch-resnet18-train/fold0_best_score.pth'

model.load_state_dict(torch.load(weights_path))
from tqdm import tqdm



with timer('inference'):

    

    model.to(device) 

    

    preds = np.zeros((len(test_dataset)))



    for i, images in tqdm(enumerate(test_loader)):

            

        images = images.to(device)

            

        with torch.no_grad():

            y_preds = model(images)

            

        preds[i * batch_size: (i+1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()

test_df['preds'] = preds.astype(int)

submission = sample_submission.merge(test_df.rename(columns={'id': 'Id'})[['Id', 'preds']], on='Id').drop(columns='Predicted')

submission['Predicted'] = submission['preds'].map(class_map)

submission = submission.drop(columns='preds')

submission.to_csv('submission.csv', index=False)

submission.head()