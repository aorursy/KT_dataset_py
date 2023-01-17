# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.metrics import error_rate

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import numpy as np 

import zipfile



import matplotlib.pyplot as plt

import time

import os
from fastai.callbacks import ActivationStats

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
classes = ['Boletus','Entoloma','Russula','Suillus','Lactarius','Amanita','Agaricus','Hygrocybe','Cortinarius']
bs = 128
path = Path('/kaggle/input')

dest = path

dest.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.35,

                                  ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
learn.fit_one_cycle(8, wd=0.9)
learn.unfreeze()
learn.lr_find(start_lr = slice(1e-5),end_lr=slice(1))

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-5, 2e-3), pct_start=0.8, wd=0.9)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-8), pct_start=0.8, wd=0.9)
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
preds_test,y_test, losses_test= learn.get_preds(ds_type=data.test_ds, with_loss=True)
print("Accuracy on test set: ", accuracy(preds_test,y_test).item())