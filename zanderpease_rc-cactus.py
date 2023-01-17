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



from pathlib import Path

from fastai import *

from fastai.vision import *

import torch



path = Path("../input/aerial-cactus-identification")

model_path = Path("../input/resnet34")
!mkdir -p /tmp/.torch/models

!cp /kaggle/input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
test_path = path/'test'

test_path.ls()
train_csv = pd.read_csv(path/'train.csv')



test_df = pd.read_csv(path/'sample_submission.csv')

test_data = ImageList.from_df(test_df, path=path/'test', folder='test')



# ImageDataBunch automatically splits into training and validation sets

data = ImageDataBunch.from_csv(path, csv_labels='train.csv', folder='train/train', size=32)



data.add_test(test_data)



data.normalize(imagenet_stats)



data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=model_path)
learn.fit_one_cycle(4)
learn.save('first')
learn.lr_find()

learn.recorder.plot()
# Fine-tune

learn.load('first')

learn.unfreeze()

learn.fit_one_cycle(3, max_lr=5e-5)
# View top losses

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
# Generate predictions

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# for v in preds.numpy()[:, 0]:

#     print(v)
# Generate submission

test_df = pd.read_csv(path/'sample_submission.csv')

test_df.has_cactus = preds.numpy()[:, 0]

test_df.to_csv('submission.csv', index=False)
# bin = preds[:, 0]>0.5

# foo = 0

# for b in bin:

#     if b:

#         foo += 1

# foo
# test_df = pd.read_csv(path/'sample_submission.csv')

# test_df.head()