# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *

PATH = Path('../input')

TRN = PATH/'train'

TEST = PATH/'test'
df = pd.read_csv(PATH/'train.csv')

df.head()
import seaborn as sns

sns.countplot(df.has_cactus)
tfms = get_transforms()

data = ImageDataBunch.from_csv(path=PATH, folder='train/train', test='test/test',  csv_labels='train.csv', ds_tfms=tfms, bs=64, valid_pct=0.2, num_workers=1).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,7),ds_type=DatasetType.Train)
data.path = Path('.')

learner = cnn_learner(data, models.resnet34, metrics=accuracy)
learner.lr_find()
learner.recorder.plot()
learner.fit(1)
learner.unfreeze()
learner.fit_one_cycle(2, max_lr=slice(1e-3, 1e-1))
preds, _ = learner.get_preds(ds_type=DatasetType.Test)

preds
preds = torch.argmax(preds, dim=1)
df = pd.read_csv(PATH/'sample_submission.csv')
df.head()
df.has_cactus = preds

df.head()
df.to_csv('submission.csv', index=False)