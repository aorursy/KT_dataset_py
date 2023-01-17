import os

import json

import sys

import gc

import random

import time

from typing import Dict

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

from torch.nn import functional as F

import torchvision.models as M

from torch.optim import Adam, SGD

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset

import torchvision.models as models



from albumentations import Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip

from albumentations.pytorch import ToTensorV2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
submission = pd.read_csv('../input/imet-2020-fgvc7/sample_submission.csv')
# Utils

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

            #Resize(HEIGHT, WIDTH),

            RandomCrop(256, 256),

            HorizontalFlip(p=0.5),

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

            

        ])
batch_size = 128



test_dataset = TestDataset(submission, transform=get_transforms(data='valid'))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class AvgPool(nn.Module):

    def forward(self, x):

        return F.avg_pool2d(x, x.shape[2:])



class ResNet(nn.Module):

    def __init__(self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False):

        super().__init__()

        self.net = net_cls(pretrained=pretrained)

        self.net.avgpool = AvgPool()

        if dropout:

            self.net.fc = nn.Sequential(

                nn.Dropout(),

                nn.Linear(self.net.fc.in_features, num_classes),

            )

        else:

            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)



    def fresh_params(self):

        return self.net.fc.parameters()



    def forward(self, x):

        return self.net(x)





class DenseNet(nn.Module):

    def __init__(self, num_classes, pretrained=False, net_cls=M.densenet121):

        super().__init__()

        self.net = net_cls(pretrained=pretrained)

        self.avg_pool = AvgPool()

        self.net.classifier = nn.Linear(

            self.net.classifier.in_features, num_classes)



    def fresh_params(self):

        return self.net.classifier.parameters()



    def forward(self, x):

        out = self.net.features(x)

        out = F.relu(out, inplace=True)

        out = self.avg_pool(out).view(out.size(0), -1)

        out = self.net.classifier(out)

        return out



class ShuffleNet(nn.Module):

    def __init__(self, num_classes, pretrained=False, net_cls=M.shufflenet_v2_x1_0, dropout=False):

        super().__init__()

        self.net = net_cls(pretrained=pretrained)

        if dropout:

            self.net.fc = nn.Sequential(

                nn.Dropout(),

                nn.Linear(self.net.fc.in_features, num_classes),

            )

        else:

            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)



    def fresh_params(self):

        return self.net.classifier.parameters()



    def forward(self, x):

        return self.net(x)

    

class SqueezeNet(nn.Module):

    def __init__(self, num_classes, pretrained=False, net_cls=M.squeezenet1_0, dropout=False):

        super().__init__()

        self.net = net_cls(pretrained=pretrained)

    

        fin_conv = nn.Conv2d(512, num_classes, kernel_size=1)

        self.net.classifier = nn.Sequential(

            nn.Dropout(p=0.5),

            fin_conv,

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))

        )

    def fresh_params(self):

        return self.net.classifier.parameters()



    def forward(self, x):

        out = self.net.features(x)

        out = self.net.classifier(out)

        out = out.view(-1, 3474)

        return out

    

    

class MobileNet(nn.Module):

    def __init__(self, num_classes, pretrained=False, net_cls=M.shufflenet_v2_x1_0, dropout=False):

        super().__init__()

        self.net = net_cls(pretrained=pretrained)

        

        self.net.classifier = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(self.net.last_channel, num_classes),

        )



    def fresh_params(self):

        return self.net.classifier.parameters()



    def forward(self, x):

        return self.net(x)

        



resnet50 = partial(ResNet, net_cls=M.resnet50)

densenet121 = partial(DenseNet, net_cls=M.densenet121)

shufflenet = partial(ShuffleNet, net_cls=M.shufflenet_v2_x1_0)

squeezenet = partial(SqueezeNet, net_cls=M.squeezenet1_0)

mobilenet = partial(MobileNet, net_cls=M.mobilenet_v2)
from efficientnet_pytorch import * 



def load_pretrained_weights2(model, model_name, weights_path=None, load_fc=True, advprop=False):

    """Loads pretrained weights from weights path or download using url.

    Args:

        model (Module): The whole model of efficientnet.

        model_name (str): Model name of efficientnet.

        weights_path (None or str): 

            str: path to pretrained weights file on the local disk.

            None: use pretrained weights downloaded from the Internet.

        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.

        advprop (bool): Whether to load pretrained weights

                        trained with advprop (valid when weights_path is None).

    """

    if isinstance(weights_path,str):

        if torch.cuda.is_available():

            state_dict = torch.load(weights_path)

        else:

            state_dict = torch.load(weights_path,map_location=torch.device('cpu'))

    else:

        # AutoAugment or Advprop (different preprocessing)

        url_map_ = url_map_advprop if advprop else url_map

        state_dict = model_zoo.load_url(url_map_[model_name])

    

    if load_fc:

        ret = model.load_state_dict(state_dict, strict=False)

        assert not ret.missing_keys, f'Missing keys when loading pretrained weights: {ret.missing_keys}'

    else:

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        ret = model.load_state_dict(state_dict, strict=False)

        assert set(ret.missing_keys) == set(

            ['_fc.weight', '_fc.bias']), f'Missing keys when loading pretrained weights: {ret.missing_keys}'

    assert not ret.unexpected_keys, f'Missing keys when loading pretrained weights: {ret.unexpected_keys}'



    print('Loaded pretrained weights for {}'.format(model_name))
DIR_WEIGHTS = '/kaggle/input/imet2020'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/EfficientNet_b0_epoch5.pth'



# model = resnet50(num_classes=N_CLASSES, pretrained=False)



# model = squeezenet(num_classes=N_CLASSES, pretrained=False)



# model = densenet121(num_classes=N_CLASSES)



# model = mobilenet(num_classes=N_CLASSES, pretrained=False)



# model = shufflenet(num_classes=N_CLASSES)



model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 3474})

load_pretrained_weights2(model, 'efficientnet-b0', weights_path=WEIGHTS_FILE, advprop=False)



# if torch.cuda.is_available():

#     model.load_state_dict(torch.load(WEIGHTS_FILE))

# else:

#     model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))

    

model.to(device)
criterion = nn.BCEWithLogitsLoss(reduction='none')
with timer('inference'):

    

    model.eval()

    

    preds = []

    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))



    for i, images in tk0:

            

        images = images.to(device)

            

        with torch.no_grad():

            y_preds = model(images)

            

        preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
threshold = 0.10

predictions = np.concatenate(preds) > threshold



for i, row in enumerate(predictions):

    ids = np.nonzero(row)[0]

    submission.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])

    

submission.to_csv('submission.csv', index=False)

submission.head()