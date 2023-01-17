import pandas as pd

pd.options.display.max_columns=999

pd.options.display.max_rows=999

from collections import Counter, defaultdict

import re

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objects as go

%reload_ext autoreload

%autoreload 2

from tqdm import tqdm_notebook, trange

import multiprocessing as mp

import pickle

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from glob import glob

from itertools import chain

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix,precision_recall_curve

import datetime

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, ConcatDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from copy import deepcopy

import pandas_profiling

import math

import os

from pytorch_wrapper_local.container import *

from pytorch_wrapper_local.callback import *

from functools import partial

from lightgbm import LGBMRegressor, LGBMClassifier

from utility_cto_ds.utility_prepare import *

from utility_cto_ds.utility_explore import *

from utility_cto_ds.utility_model_nn import *

from ipywidgets import interact

import ipywidgets as widgets

import warnings

warnings.filterwarnings('ignore')

from pathlib import Path



def extract_uppercase(ser: pd.Series):

    df = ser.str.extract('(^[A-Z][A-Z\s]+)')

    return df[0].str.replace('\s[A-Z]$', '').str.strip()



def flatten_col_names(df):

    df.columns = [df.columns.levels[0][c0] + '_' + df.columns.levels[1][c1]  for c0,c1 in zip(*df.columns.codes)]

    

txn_df['text_clean'] = extract_uppercase(txn_df.t0)

txn_df['text_clean'] = txn_df['text_clean'].fillna('xxna')



num_cols = ['n4','n7']

cat_cols = ['c5','c6','c7','n3','n5','n6','text_clean']



txn_df = otherify(txn_df,cat_cols,th=.01)

txn_df = otherify(txn_df,[cat_cols[-1]],th=.001)



# create diff to cat feat

for num_col in num_cols:

    for cat_col in cat_cols:

        txn_df[f'{num_col}_{cat_col}_group_mean_diff'] = txn_df[num_col] - txn_df.groupby(cat)[num_col].transform('mean')



num_col_aug = [f'{num_col}_{cat_col}_group_mean_diff' for num_col in num_cols for cat_col in cat_cols]



# create numerical stat by id

txn_num_agg = txn_df.groupby('id')[num_cols+num_col_aug].agg(agg_feats[:-3])



flatten_col_names(txn_num_agg)



for col in cat_cols: txn_df[col] = txn_df[col].astype(str)

# count vectorize categorical features

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_one_hot = ohe.fit_transform(txn_df[cat_cols])



cat_one_hot_df = pd.DataFrame(data=cat_one_hot,index=txn_df.id.values)



cat_one_hot_df['id'] = txn_df.id.values



cat_one_hot_sum = cat_one_hot_df.groupby('id').sum()



cat_one_hot_sum = cat_one_hot_sum.reset_index()



# one hot encode old cc label and sum

old_cc_label = txn_df.drop_duplicates(subset=['id','old_cc_no','old_cc_label'])



ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

old_cc_one_hot = ohe.fit_transform(old_cc_label.old_cc_label.values.reshape(-1,1))



old_cc_one_hot_df = pd.DataFrame(data=old_cc_one_hot)

old_cc_one_hot_df['id'] = old_cc_label.id.values



old_cc_one_hot_sum = old_cc_one_hot_df.groupby('id').sum()



old_cc_one_hot_sum = old_cc_one_hot_sum.reset_index()



demo_cat_cols = ['c0','c1','c2','c3','c4','n0','n2']

demo_df_cat = demo_df.set_index('id')[demo_cat_cols]



# fix missing

demo_df_no_miss = fix_missing(demo_df_cat,num_cols=None,cat_cols=demo_cat_cols)



# otherify

demo_df_no_miss = otherify(demo_df_no_miss,demo_cat_cols)



# encode df

demo_encoder, demo_decoder = get_enc_dec_from_cols(demo_df_no_miss, cols=demo_cat_cols)



demo_df_enc = encode_df(demo_df_no_miss,demo_encoder)



demo_df_enc = demo_df_enc.reset_index()



tmp_train, tmp_valid = train_test_split(train_df, test_size=0.2, stratify=train_df.label)



train_id = tmp_train.id.values

valid_id = tmp_valid.id.values

test_id = test_df.id.values



# numeric agg

txn_num_agg_train = txn_num_agg[txn_num_agg.id.isin(train_id)].set_index('id')

txn_num_agg_valid = txn_num_agg[txn_num_agg.id.isin(valid_id)].set_index('id')

txn_num_agg_test = txn_num_agg[txn_num_agg.id.isin(test_id)].set_index('id')



txn_num_agg_train = txn_num_agg_train.add_prefix('txn_num_agg')

txn_num_agg_valid = txn_num_agg_valid.add_prefix('txn_num_agg')

txn_num_agg_test = txn_num_agg_test.add_prefix('txn_num_agg')



# old label one hot

old_cc_one_hot_sum_train = old_cc_one_hot_sum[old_cc_one_hot_sum.id.isin(train_id)].set_index('id')

old_cc_one_hot_sum_valid = old_cc_one_hot_sum[old_cc_one_hot_sum.id.isin(valid_id)].set_index('id')

old_cc_one_hot_sum_test = old_cc_one_hot_sum[old_cc_one_hot_sum.id.isin(test_id)].set_index('id')



old_cc_one_hot_sum_train = old_cc_one_hot_sum_train.add_prefix('old_cc_oh_')

old_cc_one_hot_sum_valid = old_cc_one_hot_sum_valid.add_prefix('old_cc_oh_')

old_cc_one_hot_sum_test = old_cc_one_hot_sum_test.add_prefix('old_cc_oh_')



# concat numeric feature

sta_num_train = pd.concat([txn_num_agg_train,old_cc_one_hot_sum_train],1).fillna(0)

sta_num_valid = pd.concat([txn_num_agg_valid,old_cc_one_hot_sum_valid],1).fillna(0)

sta_num_test = pd.concat([txn_num_agg_test,old_cc_one_hot_sum_test],1).fillna(0)



# scale

scaler = StandardScaler()

sta_num_train.iloc[:,:] = scaler.fit_transform(sta_num_train)

sta_num_valid.iloc[:,:] = scaler.transform(sta_num_valid)

sta_num_test.iloc[:,:] = scaler.transform(sta_num_test)



# demo

demo_train = demo_df_enc[demo_df_enc.id.isin(train_id)].set_index('id')

demo_valid = demo_df_enc[demo_df_enc.id.isin(valid_id)].set_index('id')

demo_test = demo_df_enc[demo_df_enc.id.isin(test_id)].set_index('id')



demo_train = demo_train.add_prefix('demo_')

demo_valid = demo_valid.add_prefix('demo_')

demo_test = demo_test.add_prefix('demo_')



# align feature together

feat_train = pd.concat([sta_num_train,demo_train],1).fillna(0)

feat_valid = pd.concat([sta_num_valid,demo_valid],1).fillna(0)

feat_test = pd.concat([sta_num_test,demo_test],1).fillna(0)



# align test feature with test id

feat_test = feat_test.loc[test_df.id.values]



train_df = train_df.set_index('id')



sta_num_cols = [col for col in feat_train.columns if not 'demo_' in col]

sta_cat_cols = [col for col in feat_train.columns if 'demo_' in col]



sta_num_train = feat_train[sta_num_cols]

sta_num_valid = feat_valid[sta_num_cols]

sta_num_test = feat_test[sta_num_cols]



sta_cat_train = feat_train[sta_cat_cols]

sta_cat_valid = feat_valid[sta_cat_cols]

sta_cat_test = feat_test[sta_cat_cols]



# align train/valid target with feature

targ_train = train_df.loc[feat_train.index].values.reshape(-1)

targ_valid = train_df.loc[feat_valid.index].values.reshape(-1)



############################## NEURAL NET ##############################

emb_ins = [max(demo_encoder[col].values())+1 for col in demo_cat_cols]

static_num_size = sta_num_train.shape[1]

c = 13



emb_ins, static_num_size



class_counter = Counter(targ_train.reshape(-1))



# get class weight

class_cnt = torch.tensor([class_counter[c] for c in range(c)],dtype=torch.float)



class_w = class_cnt/class_cnt.sum()



def make_one_hot(labels, C=13):

    one_hot_targ = torch.zeros(len(labels), C).cuda()

    one_hot_targ[range(len(labels)),labels] = 1

    return one_hot_targ



class FocalLossMultiLabel(nn.Module):

    def __init__(self, gamma, weight):

        super().__init__()

        self.gamma = gamma

        self.log_softmax = nn.LogSoftmax()

        self.nll = nn.NLLLoss(weight=weight, reduce=False)



    def forward(self, input, target):

        input = self.log_softmax(input)

        loss = self.nll(input, target)

        one_hot = make_one_hot(target.unsqueeze(dim=1))

        inv_probs = 1 - input.exp()

        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma

        loss = loss * focal_weights

        

        return loss.mean()    

    

in_dtypes = [torch.float, torch.long]

targ_dtypes = [torch.long,torch.long]



train_ds = UniversalDataset(ins=[sta_num_train.values, sta_cat_train.values],

              in_dtypes=in_dtypes,

              targs=[targ_train,targ_train],

              targ_dtypes=targ_dtypes)



valid_ds = UniversalDataset(ins=[sta_num_valid.values, sta_cat_valid.values],

                           in_dtypes=in_dtypes,

                           targs=[targ_valid,targ_valid],

                           targ_dtypes=targ_dtypes)



test_ds = UniversalDataset(ins=[sta_num_test.values, sta_cat_test.values],

                          in_dtypes=in_dtypes,

                          targs = [None,None],

                          targ_dtypes = targ_dtypes)



bs = 1024

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size= min(bs,len(valid_ds)), shuffle=False)

test_dl = DataLoader(test_ds, batch_size= min(bs,len(test_ds)), shuffle=False)



data_train = DataBunch(train_dl, valid_dl)



# stack train & valid data then retrain on whole data

all_ds = ConcatDataset([train_ds,valid_ds])

all_dl = DataLoader(all_ds,batch_size=bs, shuffle=True)

all_dl_ = DataLoader(all_ds,batch_size=bs, shuffle = False)

data_all = DataBunch(all_dl, None)



model = get_basenn(static_cat_ins = emb_ins, emb_p = 0.05, static_num_ins = static_num_size,

                   fc_sizes = [400,200], fc_ps = [0.5,0.25],

                   out_tasks = ['classification','classification'],

                   out_ranges=[None,None],

                   out_features=[c,c])

model.to(model.device)

cbfs = [partial(AvgStatsCallback,None)]

loss_func = MultitasksLoss([nn.CrossEntropyLoss(class_w).cuda(),FocalLossMultiLabel(.5,class_w).cuda()],

                           loss_weights=[.8,.2])

learner = Learner(model,data_train,loss_func,lr=1e-3,optim_name='sgd',cb_funcs=cbfs)

learner.fit1cycle(30,5e-1)



preds_test = extract_pred_outs(test_dl, model)



pred_softmax = numpy_softmax(preds_test[0])

pred_max_prob = np.max(pred_softmax,1)

pred_index = np.argmax(pred_softmax,1)



# condition

cond = pred_max_prob>0.5

semi_id = test_id[cond]

semi_targ = pred_index[cond]



# get corresponding features

semi_ds = UniversalDataset(ins=[sta_num_test.values[cond], sta_cat_test.values[cond]],

                          in_dtypes=in_dtypes,

                          targs = [None,None],

                          targ_dtypes = targ_dtypes)



aug_all_ds = ConcatDataset([all_ds,semi_ds])

aug_all_dl = DataLoader(aug_all_ds,batch_size=bs, shuffle=True)

data_all_aug = DataBunch(aug_all_dl, None)



model = get_basenn(static_cat_ins = emb_ins, emb_p = 0.05, static_num_ins = static_num_size,

                   fc_sizes = [400,200], fc_ps = [0.5,0.25],

                   out_tasks = ['classification','classification'],

                   out_ranges=[None,None],

                   out_features=[c,c])

model.to(model.device)

cbfs = [partial(AvgStatsCallback,None)]

loss_func = MultitasksLoss([nn.CrossEntropyLoss(class_w).cuda(),FocalLossMultiLabel(.5,class_w).cuda()],

                           loss_weights=[.8,.2])

learner = Learner(model,data_all_aug,loss_func,lr=1e-3,optim_name='sgd',cb_funcs=cbfs)

learner.fit1cycle(30,5e-1)



preds_test = extract_pred_outs(test_dl, model)



pred_softmax = numpy_softmax(preds_test[0])



############################## LGB ##############################

train_list = [sta_num_train, sta_cat_train]

valid_list = [sta_num_valid, sta_cat_valid]

test_list = [sta_num_test, sta_cat_test]

suffices = ['static_num','static_cat']

feat_names = [suffix + f'_{i}' for suffix,feat in zip(suffices,train_list) for i in range(feat.shape[1])]



train_df = pd.DataFrame(data=np.concatenate(train_list,1),columns=feat_names)

valid_df = pd.DataFrame(data=np.concatenate(valid_list,1),columns=feat_names)

test_df_ = pd.DataFrame(data=np.concatenate(test_list,1),columns=feat_names)



train_df['targ'] = targ_train

valid_df['targ'] = targ_valid



df = pd.concat([train_df,valid_df],0).reset_index(drop=True)



class_w_dict = {k:v for k,v in enumerate(class_w.numpy())}



lgb = LGBMClassifier(objective='cross_entropy',n_estimators=1000,class_weight=class_w_dict)

lgb.fit(df.drop('targ',axis=1),df.targ)

pred_prob = lgb.predict_proba(test_df_)



pred_ens = (pred_prob + pred_softmax)

pred_ens = pred_ens/pred_ens.sum(1).reshape(-1,1)



submit_df = pd.DataFrame(data=pred_ens,columns=[f'class{i}' for i in range(13)])

submit_df['id'] = test_df.id

submit_df['id'] = submit_df['id'].astype(int)

submit_df = submit_df[['id'] + [f'class{i}' for i in range(13)]].fillna(1)



submit_df.to_csv('submission.csv',index=False)