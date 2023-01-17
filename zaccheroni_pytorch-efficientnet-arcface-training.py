!mkdir -p /tmp/pip/cache/
!cp ../input/resources-for-google-landmark-recognition-2020/efficientnet_pytorch-0.6.3-py3-none-any.whl /tmp/pip/cache/
!pip install --no-index --find-links /tmp/pip/cache/ efficientnet_pytorch
import efficientnet_pytorch
# if you need notification
#!pip install slackweb
# import slackweb
import os
import gc
gc.enable()
import sys
import math
import json
import time
import random
from glob import glob
from datetime import datetime

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision
from torch import Tensor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import albumentations as A

import sklearn

import warnings
warnings.filterwarnings("ignore")

import csv
import pprint
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from sklearn.model_selection import train_test_split
import torch.optim as optim
# Seed everything to avoid non-determinism.
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
seed_everything()
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MIN_SAMPLES_PER_CLASS = 30
BATCH_SIZE = 64
NUM_WORKERS = multiprocessing.cpu_count()
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 8
LOG_FREQ = 400
NUM_TOP_PREDICTS = 20
train = pd.read_csv('../input/landmark-recognition-2020/train.csv')
test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
train_dir = '../input/landmark-recognition-2020/train/'
test_dir = '../input/landmark-recognition-2020/test/'
train, val = train_test_split(train, test_size=0.02)
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir:str, mode: str):
        self.df = dataframe
        self.mode = mode
        self.image_dir = image_dir
        
        transforms_list = []
        if self.mode == 'train':
            # Increase image size from (64,64) to higher resolution,
            # Make sure to change in RandomResizedCrop as well.
            transforms_list = [
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(64),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ]
        else:
            transforms_list.extend([
                # Keep this resize same as train
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int):
        image_id = self.df.iloc[index].id
        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        image = Image.open(image_path)
        image = self.transforms(image)

        if self.mode == 'test':
            return {'image':image}
        else:
            return {'image':image, 
                    'target':self.df.iloc[index].landmark_id}

    def __len__(self) -> int:
        return self.df.shape[0]
def load_data(train, val, test, train_dir, test_dir):
    counts = train.landmark_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)
    all_val_count = val.shape[0]

    train = train.loc[train.landmark_id.isin(selected_classes)]
    val = val.loc[val.landmark_id.isin(selected_classes)]
    print('train_df', train.shape)
    print('val_df', val.shape)
    print('test_df', test.shape)

    # filter non-existing test images
    exists = lambda img: os.path.exists(f'{test_dir}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')
    test = test.loc[test.id.apply(exists)]
    print('test_df after filtering', test.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train.landmark_id = label_encoder.transform(train.landmark_id)
    val.landmark_id = label_encoder.transform(val.landmark_id)

    train_dataset = ImageDataset(train, train_dir, mode='train')
    val_dataset = ImageDataset(val, train_dir, mode='train')
    test_dataset = ImageDataset(test, test_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, label_encoder, num_classes, all_val_count
def adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters,
                      lr=lr,
                      betas=betas,
                      eps=eps,
                      weight_decay=weight_decay)
class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, train, label=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if train:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.cuda().view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        else:
            output = cosine
        output *= self.s

        return output
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.gem = GeM()
        self.output_filter = self.base._fc.in_features
        self.fc = nn.Linear(self.output_filter, 1000)
        self.arcface = ArcMarginProduct(1000, num_classes)
    def forward(self, x, label):
        x = self.base.extract_features(x)
        x = self.gem(x).squeeze()
        x = self.fc(x)
        if self.training:
            x = self.arcface(x, self.training, label)
        else:
            x = self.arcface(x, self.training)
        return x
def val_step(val_loader,
        model,
        criterion,
        label_encoder,
        all_val_count):
    
    val_losses = AverageMeter()
    val_gap_score = AverageMeter()
    val_acc = AverageMeter()
    model.eval()
    acc_count = 0
    first = True
    end = time.time()
    for i, data in enumerate(val_loader):
        input_ = data['image']
        target = data['target']
        batch_size, _, _, _ = input_.shape
        
        output = model(input_.cuda(), target.cuda())
        confs, predicts = torch.max(output.detach(), dim=1)
        
        if first:
            all_confs = confs
            all_predicts = predicts
            all_targets = target
            first = False
        else:
            all_confs = torch.cat([all_confs, confs])
            all_predicts = torch.cat([all_predicts, predicts])
            all_targets = torch.cat([all_targets, target])

    val_gap_score = GAP(all_predicts, all_confs, all_targets)
    val_gap_score = val_gap_score * len(all_confs) / all_val_count
    
    for i, (c, p, t) in enumerate(zip(all_confs, all_predicts, all_targets)):
        if p == t:
            acc_count += 1
                
    acc = float(acc_count) / all_val_count
    val_time = time.time() - end
    return acc, val_gap_score, val_time
def train_step(train_loader, 
          model, 
          criterion, 
          optimizer,
          epoch, 
          lr_scheduler):
    print(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    print(f'total batches: {num_steps}')

    end = time.time()
    lr = None

    for i, data in enumerate(tqdm(train_loader)):
        input_ = data['image']
        target = data['target']
        batch_size, _, _, _ = input_.shape
        
        output = model(input_.cuda(), target.cuda())
        loss = criterion(output, target.cuda())
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (num_steps//10) == 0:
            acc, val_gap, val_time = val_step(val_loader, model, criterion, label_encoder, all_val_count)
            print('validation time '+str(val_time))
            print(f'{epoch} [{i}/{num_steps}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})\t'
                    f'val_acc {acc}\t'
                    f'val_GAP {val_gap:.4f}\t'
                 )
            #slack = slackweb.Slack(url="~~~~~~")
            #slack.notify(text= f'{epoch} [{i}/{num_steps}]\t'
                    #f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    #f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})\t'
                    #f'val_acc {acc}\t'
                    #f'val_GAP {val_gap:.4f}\t')

    print(f' * average GAP on train {avg_score.avg:.4f}')
    print(f' time {batch_time.sum:.4f}')
    return avg_score.avg, losses.avg
def inference(data_loader, model):
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if data_loader.dataset.mode != 'test':
                input_, target = data['image'], data['target']
            else:
                input_, target = data['image'], None

            output = model(input_.cuda())
            output = activation(output)

            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets
if __name__ == '__main__':
    modelname = 'the_model'
    input_dir = '../input/'
    
    global_start_time = time.time()
    train_loader, val_loader, test_loader, label_encoder, num_classes, all_val_count = load_data(train, val, test, train_dir, test_dir)

    all_classes = label_encoder.classes_
    all_classes = list(all_classes)
    selected_classes = train.landmark_id
    with open('selected_classes.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(all_classes)
    
    model = EfficientNetEncoderHead(depth=7, num_classes=num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS, eta_min=1e-6)

    s = False
    # if there is 'learning.txt', read it and start training from the epoch which is written in that file.
    if os.path.exists(input_dir + 'learning.txt'):
        with open(input_dir + 'learning.txt') as f:
            s = f.read()
            
    # opttimizer saving dir
    opt_shc_path = 'optimizer_and_scheduler'
        
    
    if s:
        model.load_state_dict(torch.load(input_dir + 'the_model'+s+'.pth'))
        start_epoch = int(s) + 1
        checkpoint = torch.load(input_dir + 'optimizer_and_scheduler')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('optimizer and scheduler are loaded')
        
        pre_history = pd.read_csv(input_dir + 'the_model_history.csv')
    else:
        pre_history = pd.DataFrame(columns=['epoch', 'GAP', 'loss'])
        start_epoch = 1

        
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print('-' * 50)
        score, loss = train_step(train_loader, model, criterion, optimizer, epoch, scheduler)
        pre_history = pre_history.append({'GAP':score,'epoch':epoch,'loss':loss}, ignore_index=True)
        
        model_path = 'the_model'+str(epoch)+'.pth'
        state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(model.state_dict(), model_path)
        torch.save(state, opt_shc_path)
        
        with open('learning.txt', mode='w') as f:
            f.write(str(epoch))
            
        acc, val_gap, _ = val_step(val_loader, model, criterion, label_encoder, all_val_count)
        
        pre_history.to_csv('the_model_history.csv')
            
        # if you want to know about learning on slack
        #slack = slackweb.Slack(url="~~~")
        #slack.notify(text= f'{epoch:.4f} epoch finished\t'
        #            f'val_acc {acc}\t'
        #            f'val_GAP {val_gap:.4f}\t')
        
    #slack.notify(text= 'all learning finished')