1
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev > /dev/null



!pip install -q --upgrade efficientnet-pytorch

!pip install -q --upgrade pytorchcv
from glob import glob

from sklearn.model_selection import GroupKFold

import cv2

from skimage import io

import torch

import torch.nn.functional as F

from torch import nn

import os

from datetime import datetime

import time

import random

from tqdm import tqdm

import cv2

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

import sklearn

import warnings

from pytorchcv.model_provider import get_model as ptcv_get_model

from efficientnet_pytorch import EfficientNet

from sklearn import metrics



import warnings  

warnings.filterwarnings('ignore')
import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils





from sklearn.model_selection import train_test_split



from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup



import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm



from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

from torch.optim import lr_scheduler





import sys



import gc

import os

import random



import skimage.io

import cv2

from PIL import Image

import numpy as np

import pandas as pd

import scipy as sp



import sklearn.metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold



from functools import partial



from torch.utils.data import DataLoader, Dataset

import torchvision.models as models



from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip

from albumentations.pytorch import ToTensorV2





from contextlib import contextmanager

from pathlib import Path

from collections import defaultdict, Counter



warnings.filterwarnings("ignore")
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    

seed_everything(seed=42)
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"

DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'

fold = 0



dataset_train = pd.read_csv('../input/alaska2split/fold_{}_train.csv'.format(fold))# .iloc[:1000, :]

dataset_val = pd.read_csv('../input/alaska2split/fold_{}_valid.csv'.format(fold))# .iloc[:1000, :]

sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
def onehot(size, target):

    vec = torch.zeros(size, dtype=torch.float32)

    vec[target] = 1.

    return vec



# https://www.kaggle.com/shonenkov/train-inference-gpu-baseline

class ALASKA2Dataset(Dataset):



    def __init__(self, kinds, image_names, labels, transforms=None):

        super().__init__()

        self.kinds = kinds

        self.image_names = image_names

        self.labels = labels

        self.transforms = transforms



    def __getitem__(self, index: int):

        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']



        target = onehot(4, label)

        return image, target



    def __len__(self) -> int:

        return self.image_names.shape[0]



    def get_labels(self):

        return list(self.labels)
train_transform = A.Compose([A.HorizontalFlip(p=0.5),

                             A.VerticalFlip(p=0.5),

                             A.Transpose(p=0.5),

                             A.Resize(height=512, width=512, p=1.0),

                             ToTensorV2(p=1.0),

                             ], p=1.0)



val_transform = A.Compose([A.Resize(height=512, width=512, p=1.0),

                           ToTensorV2(p=1.0),

                           ], p=1.0)



train_dataset = ALASKA2Dataset(

    kinds=dataset_train.kind.values,

    image_names=dataset_train.image_name.values,

    labels=dataset_train.label.values,

    transforms=train_transform,

)



valid_dataset = ALASKA2Dataset(

    kinds=dataset_val.kind.values,

    image_names=dataset_val.image_name.values,

    labels=dataset_val.label.values,

    transforms=val_transform,

)
print(len(train_dataset))

print(len(valid_dataset))
class Identity(nn.Module):

    def __init__(self):

        super(Identity, self).__init__()



    def forward(self, x):

        return x





class Mish(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x):

        return x * torch.tanh(F.softplus(x))





class Alaska2Model(nn.Module):



    def __init__(self, model_name='efficientnet_b3', num_classes=4):

        super().__init__()

        if model_name.lower() == 'resnet34':

            self.backbone = ptcv_get_model("resnet34", pretrained=True)



            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)

            self.backbone.output = nn.Linear(512, num_classes)

            # self.backbone.output = nn.Sequential(nn.Linear(512, 128),

            #                                      swish(),

            #                                      nn.Dropout(p=0.5),

            #                                      nn.Linear(128, num_classes))

        elif model_name.lower() == 'efficientnet_b7':

            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')

            # self.backbone._fc = nn.Linear(2560, num_classes)



            self.backbone._fc = nn.Sequential(nn.Linear(2560, 256),

                                              Mish(),

                                              nn.Dropout(p=0.5),

                                              nn.Linear(256, num_classes))

        elif model_name.lower() == 'efficientnet_b2':

            self.backbone = EfficientNet.from_pretrained('efficientnet-b2')

            # self.backbone._fc = nn.Linear(2560, num_classes)



            self.backbone._fc = nn.Sequential(nn.Linear(1408, 256),

                                              Mish(),

                                              nn.Dropout(p=0.5),

                                              nn.Linear(256, num_classes))

        elif model_name.lower() == 'efficientnet_b3':

            self.backbone = EfficientNet.from_pretrained('efficientnet-b3')

            # self.backbone._fc = nn.Linear(2560, num_classes)



            self.backbone._fc = nn.Sequential(nn.Linear(1536, 256),

                                              Mish(),

                                              nn.Dropout(p=0.5),

                                              nn.Linear(256, num_classes))

        elif model_name.lower() == 'se_resnext101':

            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)



            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)

            self.backbone.output = nn.Sequential(nn.Linear(2048, 256),

                                                 Mish(),

                                                 nn.Dropout(p=0.5),

                                                 nn.Linear(256, num_classes))

        elif model_name.lower() == 'inceptionresnetv2':

            self.backbone = ptcv_get_model("inceptionresnetv2", pretrained=True)



            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)

            self.backbone.output = nn.Sequential(nn.Linear(1536, 128),

                                                 Mish(),

                                                 nn.Dropout(p=0.5),

                                                 nn.Linear(128, num_classes))

        elif model_name.lower() == 'pnasnet5large':

            self.backbone = ptcv_get_model("pnasnet5large", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)

            self.backbone.output = nn.Sequential(nn.Linear(4320, 512),

                                                 Mish(),

                                                 nn.Dropout(p=0.5),

                                                 nn.Linear(512, num_classes))

        else:

            raise NotImplementedError



    def forward(self, x):



        x = self.backbone(x)



        return x
model = Alaska2Model()
num_epochs = 2

NUM_EPOCH = num_epochs

from torch.optim.lr_scheduler import OneCycleLR



BATCH_SIZE = 12

#model = torchvision.models.resnext50_32x4d(pretrained=True)

#model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))

# model = EfficientNet.from_name('efficientnet-b3')



#model.avg_pool = nn.AdaptiveAvgPool2d(1)

# num_ftrs = model._fc.in_features

# model._fc = nn.Linear(num_ftrs, 4)

#model.load_state_dict(torch.load("../input/pytorch-transfer-learning-baseline/model.bin"))

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

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





def alaska_weighted_auc(y_true, y_valid):

    """

    https://www.kaggle.com/anokas/weighted-auc-metric-updated

    """

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights = [2, 1]



    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)



    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])



    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)



    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)



        if mask.sum() == 0:

            continue



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min  # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric



    return competition_metric / normalization





class RocAucMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.y_true = np.array([0, 1])

        self.y_pred = np.array([0.5, 0.5])

        self.score = 0



    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)

        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

        self.y_true = np.hstack((self.y_true, y_true))

        self.y_pred = np.hstack((self.y_pred, y_pred))

        self.score = alaska_weighted_auc(self.y_true, self.y_pred)



    @property

    def avg(self):

        return self.score





class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.0):

        super(LabelSmoothing, self).__init__()

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing



    def forward(self, x, target):

        if self.training:

            x = x.float()

            target = target.float()

            logprobs = torch.nn.functional.log_softmax(x, dim=-1)



            nll_loss = -logprobs * target

            nll_loss = nll_loss.sum(-1)



            smooth_loss = -logprobs.mean(dim=-1)



            loss = self.confidence * nll_loss + self.smoothing * smooth_loss



            return loss.mean()

        else:

            return torch.nn.functional.cross_entropy(x, target)
#https://www.kaggle.com/dhananjay3/pytorch-xla-for-tpu-with-multiprocessing

#https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59/data



def train_model():

    global train_dataset, valid_dataset

        

    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True)

    

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=0,

        drop_last=True) 

        

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        )

        

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE ,

        sampler=valid_sampler,

        shuffle=False,

        num_workers=0,

        drop_last=True)

    

    #xm.master_print(f"Train for {len(train_loader)} steps per epoch")

#     LOGGER.debug(f"Train for {len(train_loader)} steps per epoch")

    # Scale learning rate to num cores

    lr  = 0.001 * xm.xrt_world_size()



    # Get loss function, optimizer, and model

    device = xm.xla_device()



    #model = model()

    '''

    for param in model.base_model.parameters(): # freeze some layers

        param.requires_grad = False'''

    

    

    global model

    

    model = model.to(device)



    criterion = LabelSmoothing().to(device) #  BCEWithLogitsLoss

    #criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10)

    

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(

        mode='min',

        factor=0.5,

        patience=1,

        verbose=False,

        threshold=0.0001,

        threshold_mode='abs',

        cooldown=0,

        min_lr=1e-8,

        eps=1e-08

    )

    

    scheduler = SchedulerClass(optimizer, **scheduler_params)



    

    def train_loop_fn(loader):

        tracker = xm.RateTracker()

        model.train()

        

        summary_loss = AverageMeter()

        final_scores = RocAucMeter()



        xm.master_print(f"Training Start:{(time.ctime())}")

        for step, (inputs, labels) in enumerate(loader):

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            

            batch_size = inputs.shape[0]

            

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            xm.optimizer_step(optimizer)

            

            final_scores.update(labels, outputs)

            summary_loss.update(loss.detach().item(), batch_size)

            if step % 200 == 0:

                xm.master_print(f"Train Step {step}/{len(loader)} - Loss:{summary_loss.avg:0.4f} - Score:{final_scores.avg:0.4f}")

        scheduler.step(final_scores.avg)

        

#         xm.master_print(f"Train Step {step}/{len(loader)} - Loss:{summary_loss.avg:0.4f} - Score:{final_scores.avg:0.4f}")

    

    def test_loop_fn(loader):

        summary_loss = AverageMeter()

        final_scores = RocAucMeter()

        for step, (inputs, labels) in enumerate(loader):

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            batch_size = inputs.shape[0]

            outputs = model(inputs)

            

            loss = criterion(outputs, labels)

                    

            final_scores.update(labels, outputs)

            summary_loss.update(loss.detach().item(), batch_size)

            if step % 200 == 0:

                xm.master_print(f"Val Step {step}/{len(loader)} - Loss:{summary_loss.avg:0.4f} - Score:{final_scores.avg:0.4f}")        

        

        #auc_score = alaska_weighted_auc(labels.cpu().numpy(), output.cpu().numpy())

        #LOGGER.debug("auc_score according to competition metric = {} ".format(auc_score))

        #print('[xla:{}] Accuracy={:.4f}%'.format(xm.get_ordinal(), accuracy), flush=True)

        model.train()

        return final_scores.avg

    

    # Train - valid  loop

    for epoch in range(1, num_epochs + 1):

        start = time.time()

        para_loader = pl.ParallelLoader(train_loader, [device])

        train_loop_fn(para_loader.per_device_loader(device))

        

        para_loader = pl.ParallelLoader(valid_loader, [device])

        result = test_loop_fn(para_loader.per_device_loader(device))       

        

        xm.master_print("Finished training epoch {}  Val-Acc {:.4f} in {:.4f} sec".format(epoch, result,  time.time() - start))   

        if(epoch>0):

            xm.save(model.state_dict(), f"./epoch{epoch}valauc{result}.bin")

    return result
# Start training processes



def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    res = train_model()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
xm.get_xla_supported_devices()