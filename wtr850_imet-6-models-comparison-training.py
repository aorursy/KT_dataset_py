import tqdm

import random

import json

import math

import cv2

import shutil

import warnings

import numpy as np

import pandas as pd



from collections import defaultdict, Counter

from pathlib import Path

from itertools import islice

from typing import Callable, List, Dict

from PIL import Image

from functools import partial

from efficientnet_pytorch import EfficientNet



import torch

import torchvision.models as M

from torch import nn, cuda

from torch.nn import functional as F

from torch.optim import Adam

from torch.utils import model_zoo

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import (

    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,

    RandomHorizontalFlip)

from sklearn.metrics import fbeta_score

from sklearn.exceptions import UndefinedMetricWarning



N_CLASSES = 3474





DATA_ROOT = '../input/imet-2020-fgvc7/'

train = pd.read_csv('../input/imet-2020-fgvc7/train.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
folds = pd.read_csv('../input/imet2002folds/folds.csv') 

folds.head(5)
def load_image(item, root):

    image = cv2.imread(str(root + '/' + f'{item.id}.png'))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image)



def load_transform_image(item, root, image_transform, debug=False):

    image = load_image(item, root)

    image = image_transform(image)

    if debug:

        image.save('_debug.png')

    return tensor_transform(image)

class TrainDataset(Dataset):

    def __init__(self, root, df, image_transform, debug=True):

        super().__init__()

        self.root = root

        self.df = df

        self.image_transform = image_transform

        self.debug = debug



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        item = self.df.iloc[idx]



        image = load_transform_image(item, self.root, self.image_transform, debug=self.debug)

        target = torch.zeros(N_CLASSES)

        for cls in item.attribute_ids.split():

            target[int(cls)] = 1

        return image, target





class TTADataset:

    def __init__(self, root, df, image_transform, tta):

        self.root = root

        self.df = df

        self.image_transform = image_transform

        self.tta = tta



    def __len__(self):

        return len(self.df) * self.tta #tta周分する



    def __getitem__(self, idx):

        item = self.df.iloc[idx % len(self.df)]

        #print(item)

        image = load_transform_image(item, self.root, self.image_transform)

        return image, item.id
train_transform = Compose([

    RandomCrop(288),

    RandomHorizontalFlip(),

])



test_transform = Compose([

    #RandomCrop(288),

    RandomCrop(256),

    RandomHorizontalFlip(),

])



tensor_transform = Compose([

    ToTensor(),

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
folds = pd.read_csv('../input/imet2002folds/folds.csv')



train_fold = folds[folds['fold'] != 0]

valid_fold = folds[folds['fold'] == 0]
train_root = DATA_ROOT + 'train'

num_workers = 4

batch_size = 64
def make_loader(df, image_transform):

        return DataLoader(

            TrainDataset(train_root, df, image_transform, debug=0),

            shuffle=True,

            batch_size=batch_size,

            num_workers=num_workers,

        )
train_loader = make_loader(train_fold, train_transform)

valid_loader = make_loader(valid_fold, test_transform)

print(f'{len(train_loader.dataset):,} items in train, '

      f'{len(valid_loader.dataset):,} in valid')
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
#Preparation for EfficientNet

from efficientnet_pytorch import * 

DIR_WEIGHTS = '/kaggle/input/efficientnet-pytorch'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/efficientnet-b0-08094119.pth'



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

        state_dict = torch.load(weights_path)

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
# Use one of them below.



# model = resnet50(num_classes=N_CLASSES, pretrained=True)



# model = squeezenet(num_classes=N_CLASSES, pretrained=True)



# model = densenet121(num_classes=N_CLASSES, pretrained=True)



# model = mobilenet(num_classes=N_CLASSES, pretrained=True)



# model = shufflenet(num_classes=N_CLASSES, pretrained=True)



model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 3474})

load_pretrained_weights2(model, 'efficientnet-b0', weights_path=WEIGHTS_FILE, load_fc=(N_CLASSES == 1000), advprop=False)





model.to(device)
criterion = nn.BCEWithLogitsLoss(reduction='none')
def reduce_loss(loss):

    return loss.sum() / loss.shape[0]



def make_mask(argsorted, top_n):

    mask = np.zeros_like(argsorted, dtype=np.uint8)

    col_indices = argsorted[:, -top_n:].reshape(-1)

    row_indices = [i // top_n for i in range(len(col_indices))]

    mask[row_indices, col_indices] = 1

    return mask



def binarize_prediction(probabilities, threshold, argsorted=None, min_labels=1, max_labels=10):

    #3474個それぞれが当てはまるか(1)当てはまらないか(0)を出力

    assert probabilities.shape[1] == N_CLASSES

    if argsorted is None:

        argsorted = probabilities.argsort(axis=1)

    max_mask = make_mask(argsorted, max_labels)

    min_mask = make_mask(argsorted, min_labels)

    prob_mask = probabilities > threshold

    return (max_mask & prob_mask) | min_mask



def get_score(target, y_pred):

    return fbeta_score(target , y_pred, beta=2, average='samples') #βで重み付けしてるF2score?
lr = 1e-4

n_epochs = 1



optimizer = Adam(model.parameters(), lr)



model_path = 'model.pth'

best_model_path = 'best-model.pth'



valid_losses = []

best_valid_loss = float('inf')



step = 0

count = []

loss_all = []

score_all = []

for epoch in range(n_epochs):

    model.train()

    losses = []

    mean_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):

        #print(step)

        inputs = inputs.to(device)

        targets = targets.to(device)

        

        outputs = model(inputs)

        loss = reduce_loss(criterion(outputs, targets))

        batch_size = inputs.size(0)

        (batch_size*loss).backward()

        optimizer.step()

        optimizer.zero_grad()

        

        step += 1



        losses.append(loss.item())

        mean_loss = np.mean(losses[-100:])

        if step % 100 == 0:

            print(f"step:{step}/{len(train_fold)*n_epochs//batch_size}")

            print(f"mean_loss:{mean_loss}")

            

            count.append(step)

            loss_all.append(loss.item())

            

            current_score = get_score(targets.cpu().detach().numpy(), binarize_prediction(outputs.cpu().detach().numpy(), 0.1))

            score_all.append(current_score)

            print(f"current_score:{current_score}")



torch.save(model.state_dict(), model_path)

print("Finished.")
import matplotlib.pyplot as plt

#Loss

plt.title(model.__class__.__name__ +"-Loss")

plt.plot(count, loss_all, linewidth=2)

plt.grid()

plt.xlabel("Step")

plt.ylabel("Loss")

plt.show()
#Score

plt.title(model.__class__.__name__ +"-Score")

plt.plot(count, score_all, linewidth=2, color="red")

plt.grid()

plt.xlabel("Step")

plt.ylabel("Loss")

plt.show()