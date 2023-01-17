import sys

sys.version
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import torch

from torch import nn, optim

import seaborn as sns

from pathlib import Path

import PIL

import json

from fastai import *

from fastai.tabular import *

from fastai.vision import *

from fastai.metrics import error_rate

PATH = Path('content/kaggle/')

PATH

PATH = Path('../input')
train_df = pd.read_csv(PATH/'train.csv')

train_df.head()

test_df = pd.read_csv(PATH/'test.csv')

test_df.head()
ss_df = pd.read_csv(PATH/'sample_submission.csv')

ss_df.head()
train_df.describe()
test_df.describe()
dep_var = 'target'
cat_names = []
df = train_df
cont_names = []

var_counter = 0 #creating a counter

num_of_cont_vars = len(df.columns) - 2

for _ in range(num_of_cont_vars):

    name = 'var_' + str(var_counter)

    cont_names.append(name)

    var_counter+=1
procs = [FillMissing, Normalize]
valid_idx = range(len(df)-20000, len(df))
test = TabularList.from_df(test_df, path=PATH, cont_names=cont_names, procs=procs)
path = PATH



data = (TabularList.from_df(df, path=path, cont_names=cont_names, procs=procs)

        .split_by_rand_pct(valid_pct=0.1)

        .label_from_df(cols=dep_var)

        .add_test(test)

        .databunch())



print(data.train_ds.cont_names)
data.show_batch(rows= 6)
(cat_x,cont_x),y = next(iter(data.train_dl))

for o in (cat_x, cont_x, y): print(to_np(o[:5]))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
learn = tabular_learner(data, layers=[ 200 , 100], ps=[0.001,0.01], emb_drop=0.04, emb_szs={'ID_code': 20}, metrics=accuracy , path='.')

#to change to get rsqme just change accuracy to rmspe

learn.model
learn.lr_find()

learn.recorder.plot()
lr = 1e-02

learn.fit_one_cycle(7, lr , wd = 0.3)

#wd=0.2
learn.show_results()
learn.recorder.plot_losses()
learn.data.batch_size
test_preds = learn.get_preds(ds_type=DatasetType.Test)

test_preds
target_preds = test_preds[0][:,1]

test_df['target'] = target_preds
target_preds
test_df.to_csv('submission.csv', columns=['ID_code', 'target'], index=False)
sub = pd.read_csv('submission.csv')

sub.head()
preds = learn.get_preds()

pred_tensors = preds[0]

actual_labels = preds[1].numpy()
pred_tensors, actual_labels



total_to_test = 20000

correct = 0

for i in range(total_to_test):

    if(pred_tensors[i][0] > 0.5 and actual_labels[i] == 0):

        correct = correct + 1



print(f"{correct}/{total_to_test} correct")
learn.save("trained_model", return_path=True)
learn = learn.load("trained_model" )
!kaggle competitions submit -c santander-customer-transaction-prediction -f {PATH/'submission.csv'} -m "initial submission"
test_df.head()