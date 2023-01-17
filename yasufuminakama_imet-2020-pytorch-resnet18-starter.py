import os

import numpy as np 

import pandas as pd 

import json
os.listdir('../input/imet-2020-fgvc7')
train = pd.read_csv('../input/imet-2020-fgvc7/train.csv')

labels = pd.read_csv('../input/imet-2020-fgvc7/labels.csv')

submission = pd.read_csv('../input/imet-2020-fgvc7/sample_submission.csv')
train.head()
labels.head()
submission.head()
labels['attribute_name'].nunique()
from collections import Counter



cls_counts = Counter(cls for classes in train['attribute_ids'].str.split() for cls in classes)



print(len(cls_counts))
label_map = dict(labels[['attribute_id', 'attribute_name']].values.tolist())

not_in_train_labels = set(labels['attribute_id'].astype(str).values) - set(list(cls_counts))

for _id in not_in_train_labels:

    label = label_map[int(_id)]

    print(f'attribute_id: {_id}  attribute_name: {label}')
# TOP 20 common attribute

for item in sorted(cls_counts.items(), key=lambda x: x[1], reverse=True)[:20]:

    _id, count = item[0], item[1]

    label = label_map[int(_id)]

    print(f'attribute_name: {label}  count: {count}')
# Number of labels for each instance

import matplotlib.pyplot as plt



df_label_len = train.attribute_ids.str.split(" ").apply(len)

plt.figure(figsize=(25, 4))

df_label_len.value_counts().plot.bar()

plt.title(f"Number of labels for each instance")
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
def make_folds(df, n_folds, seed):

    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)

    fold_cls_counts = defaultdict(int)

    folds = [-1] * len(df)

    for item in df.sample(frac=1, random_state=seed).itertuples():

        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])

        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]

        min_count = min([count for _, count in fold_counts])

        random.seed(item.Index)

        fold = random.choice([f for f, count in fold_counts if count == min_count])

        folds[item.Index] = fold

        for cls in item.attribute_ids.split():

            fold_cls_counts[fold, cls] += 1

    df['fold'] = folds

    return df
DEBUG = False

N_FOLDS = 5

FOLD = 0



if DEBUG:

    folds = train.sample(n=10000, random_state=SEED).reset_index(drop=True).copy()

    folds = make_folds(folds, N_FOLDS, SEED)

else:

    folds = train.copy()

    folds = make_folds(folds, N_FOLDS, SEED)

    

trn_idx = folds[folds['fold'] != FOLD].index

val_idx = folds[folds['fold'] == FOLD].index

print(trn_idx.shape, val_idx.shape)
train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), 

                             folds.loc[trn_idx]['attribute_ids'], 

                             transform=get_transforms(data='train'))

valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 

                             folds.loc[val_idx]['attribute_ids'], 

                             transform=get_transforms(data='valid'))
batch_size = 128



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
model = models.resnet18(pretrained=False)

weights_path = '../input/resnet18/resnet18.pth'

model.load_state_dict(torch.load(weights_path))



model.avgpool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
from sklearn.metrics import fbeta_score





def get_score(targets, y_pred):

    return fbeta_score(targets, y_pred, beta=2, average='samples')







def binarize_prediction(probabilities, threshold: float, argsorted=None,

                        min_labels=1, max_labels=10):

    """ 

    Return matrix of 0/1 predictions, same shape as probabilities.

    """

    assert probabilities.shape[1] == N_CLASSES

    if argsorted is None:

        argsorted = probabilities.argsort(axis=1)

    max_mask = _make_mask(argsorted, max_labels)

    min_mask = _make_mask(argsorted, min_labels)

    prob_mask = probabilities > threshold

    return (max_mask & prob_mask) | min_mask





def _make_mask(argsorted, top_n: int):

    mask = np.zeros_like(argsorted, dtype=np.uint8)

    col_indices = argsorted[:, -top_n:].reshape(-1)

    row_indices = [i // top_n for i in range(len(col_indices))]

    mask[row_indices, col_indices] = 1

    return mask





def _reduce_loss(loss):

    return loss.sum() / loss.shape[0]
with timer('Train model'):

    

    n_epochs = 12

    lr = 1e-4

    

    model.to(device)

    

    optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=4, verbose=True, eps=1e-6)

    

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_score = 0.

    best_thresh = 0.

    best_loss = np.inf

    

    for epoch in range(n_epochs):

        

        start_time = time.time()



        model.train()

        avg_loss = 0.



        optimizer.zero_grad()

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))



        for i, (images, labels) in tk0:



            images = images.to(device)

            labels = labels.to(device)

            

            y_preds = model(images)

            loss = _reduce_loss(criterion(y_preds, labels))

            

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()



            avg_loss += loss.item() / len(train_loader)

            

        model.eval()

        avg_val_loss = 0.

        preds = []

        valid_labels = []

        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))



        for i, (images, labels) in tk1:

            

            images = images.to(device)

            labels = labels.to(device)

            

            with torch.no_grad():

                y_preds = model(images)

            

            preds.append(torch.sigmoid(y_preds).to('cpu').numpy())

            valid_labels.append(labels.to('cpu').numpy())



            loss = _reduce_loss(criterion(y_preds, labels))

            avg_val_loss += loss.item() / len(valid_loader)

        

        scheduler.step(avg_val_loss)

            

        preds = np.concatenate(preds)

        valid_labels = np.concatenate(valid_labels)

        argsorted = preds.argsort(axis=1)

        

        th_scores = {}

        for threshold in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:

            _score = get_score(valid_labels, binarize_prediction(preds, threshold, argsorted))

            th_scores[threshold] = _score

        

        max_kv = max(th_scores.items(), key=lambda x: x[1])

        th, score = max_kv[0], max_kv[1]



        elapsed = time.time() - start_time

        

        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

        LOGGER.debug(f'  Epoch {epoch+1} - threshold: {th}  f2_score: {score}')

        

        if score>best_score:

            best_score = score

            best_thresh = th

            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model  threshold: {best_thresh}')

            torch.save(model.state_dict(), f'fold{FOLD}_best_score.pth')

            

        if avg_val_loss<best_loss:

            best_loss = avg_val_loss

            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

            torch.save(model.state_dict(), f'fold{FOLD}_best_loss.pth')
batch_size = 128



test_dataset = TestDataset(submission, transform=get_transforms(data='valid'))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = models.resnet18(pretrained=False)

model.avgpool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, N_CLASSES)



weights_path = f'fold{FOLD}_best_score.pth'

model.load_state_dict(torch.load(weights_path))
with timer('inference'):

    

    model.to(device) 

    

    preds = []

    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))



    for i, images in tk0:

            

        images = images.to(device)

            

        with torch.no_grad():

            y_preds = model(images)

            

        preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
threshold = best_thresh

predictions = np.concatenate(preds) > threshold



for i, row in enumerate(predictions):

    ids = np.nonzero(row)[0]

    submission.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])

    

submission.to_csv('submission.csv', index=False)

submission.head()