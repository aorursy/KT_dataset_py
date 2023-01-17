import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc

import json
from pandas.io.json import json_normalize

from tqdm import trange,tqdm
from collections import Counter

pd.options.display.max_rows = 8
pd.options.display.max_columns = 999
print(os.listdir("../input"))
def read_cols(file_tag, cols):
    data_reader = pd.read_csv(f'../input/{file_tag}_v2.csv', iterator=True, chunksize=20000,
                          dtype={id_col: 'str'})
    df = pd.DataFrame()
    for it in data_reader:
        df = df.append(it[cols], ignore_index=True)
        gc.collect()
    del data_reader, it
    gc.collect()
    
    return df
id_col = 'fullVisitorId'

data_reader = pd.read_csv('../input/train_v2.csv', iterator=True, chunksize=1000, low_memory=False, 
                          dtype={id_col: 'str'})
train_df = next(data_reader)
print(train_df.shape)
train_df.info()
train_df
data_reader = pd.read_csv('../input/test_v2.csv', iterator=True, chunksize=1000, low_memory=False, 
                          dtype={id_col: 'str'})
test_df = next(data_reader)
print(test_df.shape)
test_df.info()
test_df
cols = [id_col, 'date', 'visitId', 'visitNumber', 'visitStartTime', 'channelGrouping', 'socialEngagementType']
train_df = read_cols('train', cols)
gc.collect()
train_df
test_df = read_cols('test', cols)
gc.collect()
test_df
s = train_df[id_col].astype(np.str) + '_' + train_df.visitId.astype(np.str)
print(s.value_counts())
print('-------------------------------------------------------')
s = test_df[id_col].astype(np.str) + '_' + test_df.visitId.astype(np.str)
print(s.value_counts())
df = train_df
s = df.fullVisitorId
print(s.value_counts())
print('---------------------------------')
s = df.visitId.groupby(df.fullVisitorId).nunique()
print(s.sort_values(ascending=False))
print('-------------------------------------------------------')
df = test_df
s = df.fullVisitorId
print(s.value_counts())
print('---------------------------------')
s = df.visitId.groupby(df.fullVisitorId).nunique()
print(s.sort_values(ascending=False))
df = train_df
s = df.visitId
print(s.value_counts())
print('---------------------------------')
s = df.fullVisitorId.groupby(df.visitId).nunique()
print(s.sort_values(ascending=False))
print('-------------------------------------------------------')
df = test_df
s = df.visitId
print(s.value_counts())
print('---------------------------------')
s = df.fullVisitorId.groupby(df.visitId).nunique()
print(s.sort_values(ascending=False))
s = train_df.visitNumber
print(s.describe())
print()

cnt = s.value_counts()
print(cnt)
print(cnt.loc[cnt>1].shape)
print(cnt.describe())
s = test_df.visitNumber
print(s.describe())
print()

cnt = s.value_counts()
print(cnt)
print(cnt.loc[cnt>1].shape)
print(cnt.describe())
df = train_df
s = df.visitStartTime - df.visitId
print(s.describe())
print()
print(np.sum(df.visitId != df.visitStartTime))
print(s.loc[s>0].describe())
print('-------------------------------------------')

s = pd.to_datetime(df.date.astype(np.str))
s1 = pd.to_datetime(df.visitId, unit='s')
s2 = pd.to_datetime(df.visitStartTime, unit='s')

print(np.sum(s.dt.year != s1.dt.year), np.sum(s.dt.month != s1.dt.month), np.sum(s.dt.day != s1.dt.day))
ss = (s - s1).dt.days
print(ss.min(), ss.max())
print(np.sum(s.dt.year != s2.dt.year), np.sum(s.dt.month != s2.dt.month), np.sum(s.dt.day != s2.dt.day))
ss = (s - s2).dt.days
print(ss.min(), ss.max())
df = test_df
s = df.visitStartTime - df.visitId
print(s.describe())
print()
print(np.sum(df.visitId != df.visitStartTime))
print(s.loc[s>0].describe())
print('-------------------------------------------')

s = pd.to_datetime(df.date.astype(np.str))
s1 = pd.to_datetime(df.visitId, unit='s')
s2 = pd.to_datetime(df.visitStartTime, unit='s')

print(np.sum(s.dt.year != s1.dt.year), np.sum(s.dt.month != s1.dt.month), np.sum(s.dt.day != s1.dt.day))
ss = (s - s1).dt.days
print(ss.min(), ss.max())
print(np.sum(s.dt.year != s2.dt.year), np.sum(s.dt.month != s2.dt.month), np.sum(s.dt.day != s2.dt.day))
ss = (s - s2).dt.days
print(ss.min(), ss.max())
col = 'channelGrouping'
df = train_df
cnt = df[col].value_counts()
print(cnt.shape)
print(cnt)
print('---------------------------------')
df = test_df
cnt = df[col].value_counts()
print(cnt.shape)
print(cnt)
col = 'socialEngagementType'
df = train_df
cnt = df[col].value_counts()
print(cnt.shape)
print(cnt)
print('---------------------------------')
df = test_df
cnt = df[col].value_counts()
print(cnt.shape)
print(cnt)
del train_df, test_df
gc.collect()
json_cols = ['customDimensions', 'device', 'geoNetwork', 'hits', 'totals', 'trafficSource']


def read_json_col(file_tag, col):
    data_reader = pd.read_csv(f'../input/{file_tag}_v2.csv', iterator=True, chunksize=20000,
                          converters={col: json.loads})
    s = pd.Series()
    for it in data_reader:
        s = s.append(it[col], ignore_index=True)
        gc.collect()
    del data_reader, it
    gc.collect()
    
    return s
col = 'customDimensions'
df = read_cols('train', [col])
gc.collect()
print(df[col].unique())
col = 'customDimensions'
df = read_cols('test', [col])
gc.collect()
print(df[col].unique())
col = 'device'
df = json_normalize(read_json_col('train', col))
gc.collect()
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
col = 'device'
df = json_normalize(read_json_col('test', col))
gc.collect()
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
col = 'geoNetwork'
df = json_normalize(read_json_col('train', col))
gc.collect()
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
col = 'geoNetwork'
df = json_normalize(read_json_col('test', col))
gc.collect()
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
data_reader = pd.read_csv('../input/train_v2.csv', iterator=True, chunksize=1000, low_memory=False, 
                          dtype={id_col: 'str'})
train_df = next(data_reader)
train_df.loc[100,'hits']
data_reader = pd.read_csv('../input/test_v2.csv', iterator=True, chunksize=1000, low_memory=False, 
                          dtype={id_col: 'str'})
test_df = next(data_reader)
test_df.loc[100,'hits']
col = 'totals'
df = json_normalize(read_json_col('train', col))
gc.collect()
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
dff = df.loc[df.transactions=='1']
dff = dff.fillna(-1)
gc.collect()
col = 'totalTransactionRevenue'
dff[col] = dff[col].astype(np.int64)
col = 'transactionRevenue'
dff[col] = dff[col].astype(np.int64)
gc.collect()
print(dff.loc[dff.totalTransactionRevenue - dff.transactionRevenue > 9e6].shape)
df.loc[df.transactions>'1']
col = 'totals'
df = json_normalize(read_json_col('test', col))
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
col = 'trafficSource'
df = json_normalize(read_json_col('train', col))
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
col = 'trafficSource'
df = json_normalize(read_json_col('test', col))
print(df.shape)
df
remain_cols = []
for col in df.columns:
    cnt = df[col].value_counts(dropna=False)
    if cnt.shape[0] <= 1:
        print(col)
        print(cnt.index[0])
    else:
        remain_cols.append(col)
        print(cnt)
    print('------------------------------------------------------------')
print(remain_cols)
