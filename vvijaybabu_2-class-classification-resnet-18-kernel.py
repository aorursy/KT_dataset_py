# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# This kernel draws heavily from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import argparse

import random

import shutil

import time

import warnings

import sys

import sklearn

import pandas as pd

import tqdm

import copy 



import torch

import torch.nn as nn

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.distributed as dist

import torch.optim

import torch.multiprocessing as mp

import torch.utils.data

import torch.utils.data.distributed

import torchvision.transforms as transforms

import torchvision.datasets as datasets

import torchvision.models as models

import torchvision.utils as tvutils

print("packages loaded")
#one time work of copying files suitably to the 2 class folders (cactus & nocactus)

import shutil

traindir = os.path.join('../input', 'train')

results = pd.read_csv('../input/train.csv',header = 0, index_col = 0)

print(results.shape)

os.mkdir("../data")

os.mkdir("../data/train")

os.mkdir("../data/train/nocactus")

os.mkdir("../data/train/cactus")



for i in range(results.shape[0]):

    #print("i ",i)

    if(results.iloc[i,0] == 0):

        shutil.copy(os.path.join('../input/train/train/',results.index[i]),os.path.join('../data/train/nocactus/',results.index[i]))

    else:

        shutil.copy(os.path.join('../input/train/train/',results.index[i]),os.path.join('../data/train/cactus/',results.index[i]))

        



ngpus_per_node = torch.cuda.device_count()

feature_extract = False # if true this will not re-train the model, but only change the last stage

use_pretrained = True



# create model

print("=> creating model ")

model = models.resnet18(pretrained=use_pretrained) #pretrained=use_pretrained

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = torch.nn.DataParallel(model).cuda()



if feature_extract:

        for param in model.parameters():

            param.requires_grad = False



num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)            

model.to(device)



# define loss function (criterion) and optimizer

criterion = nn.CrossEntropyLoss().cuda(ngpus_per_node)





modeldir = os.path.join('../data/', 'train')

# Data loading code

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])



#print("length", train_dataset.__len__)



#load the classification results

#now based on results, move the image to either of 2 classification bucket

#if dir "cactus" or "nocactus" is not present, create it



full_dataset = datasets.ImageFolder(

    modeldir,

    transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        normalize,

    ]))

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, {3*results.shape[0]//4,results.shape[0]- 3*results.shape[0]//4})

print("train length",train_dataset.__len__)

batch = 128

train_loader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch, shuffle= False,

    pin_memory=True)



val_loader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch, shuffle= False,

    pin_memory=True)



params_to_update = []



if feature_extract:

    for name,param in model.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

#            print("\t",name)

else:

    for name,param in model.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

#            print("\t",name)

            

optimizer = torch.optim.SGD(params_to_update, 0.001,

                            momentum=0.9,

                            weight_decay=1e-4)



import copy



best_acc1 = 0



class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):

        self.name = name

        self.fmt = fmt

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



    def __str__(self):

        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)







def accuracy(output, target,iter_cnt, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)



        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res





def train(train_loader,model, criterion, optimizer, epoch):



    # switch to train mode

    model.train()

    top1 = AverageMeter('Acc@1', ':6.2f')

    

    for i, (input,target) in enumerate(train_loader):

        #input = input.cuda(0, non_blocking=True)

        #target = target.cuda(0, non_blocking=True)

        input = input.to(device)

        target = target.to(device)

        # compute output

        output = model(input)

        loss = criterion(output, target)



        # measure accuracy and record loss

        acc1, acc5 = accuracy(output, target, i,topk=(1, 1)) #

        top1.update(acc1[0], input.size(0))



        # compute gradient and do SGD step

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()





def validate(val_loader,model, criterion):



    # switch to evaluate mode

    model.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')



    with torch.no_grad():

        end = time.time()

        for i,(input, target) in enumerate(val_loader):

            input = input.to(device)

            target = target.to(device)

            # compute output

            output = model(input)

            loss = criterion(output, target)



            # measure accuracy and record loss

            acc1, acc5 = accuracy(output, target, i,topk=(1, 1))

            top1.update(acc1[0], input.size(0))



        print(' * Acc@1 {top1.avg:.3f}'

              .format(top1=top1))



    return top1.avg



max_epoch = 15

best_model_wts = copy.deepcopy(model.state_dict())

best_acc1 = 0



for epoch in tqdm.tqdm(range(0, max_epoch)):

    train(train_loader, model, criterion, optimizer, epoch)



    # evaluate on validation set

    acc1 = validate(val_loader, model, criterion)



    # remember best acc@1 and save checkpoint

    is_best = acc1 > best_acc1

    if (acc1 > best_acc1):

        best_acc1 = acc1

        best_model_wts = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_wts)    
#now try to predict on this model

#classification: 0 = cactus, 1 = no cactus; so swap them

#load the test data

preddir = os.path.join('../input/', 'test')



test_dataset = datasets.ImageFolder(

    preddir,

    transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        normalize,

    ]))

test_count = len([name for name in os.listdir('../input/test/test/')])

test_loader = torch.utils.data.DataLoader(

    test_dataset, batch_size=1, shuffle= False,

    pin_memory=True)



# evaluate on test set

model.eval()

pred_submit = pd.DataFrame(np.zeros(test_count,dtype=int))

with torch.no_grad():

    end = time.time()

    for i,(input, target) in tqdm.tqdm(enumerate(test_loader)):

        input = input.cuda(0, non_blocking=True)

        # compute output

        output = model(input)

        val,pred = torch.max(output, 1)

        if (pred[0] == 0):

            val[0] = torch.sigmoid(val[0])

        else:

            if (pred[0] == 1):

                val[0] = 1-torch.sigmoid(val[0])

            else:

                val[0] = 2

        pred_submit.iloc[i,0] = val[0].cpu().numpy()

prediction = pd.read_csv('../input/sample_submission.csv',header = 0,index_col=0)        

pred_submit.index = prediction.index

prediction.iloc[:,0] = pred_submit.iloc[:,0]

print(prediction.head())

#os.mkdir("../output/")

prediction.to_csv("samplesubmission.csv")

 