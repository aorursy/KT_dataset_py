import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import re

import json
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import trange,tqdm
from collections import Counter

pd.options.display.max_rows = 8
pd.options.display.max_columns = 999
print(os.listdir("../input"))
data_dir = '../input/gs-flatten-data-analysis-2'
train_df_r = pd.read_pickle(os.path.join(data_dir, 'train_df_r'), compression='gzip')
print(f'train_df_r: {train_df_r.shape}')
tfs1 = pd.to_datetime('2017-05-01').timestamp()
tfe1 = pd.to_datetime('2017-10-01').timestamp()
vtfs1 = pd.to_datetime('2017-12-01').timestamp()
vtfe1 = pd.to_datetime('2018-02-01').timestamp()

tfs2 = pd.to_datetime('2018-01-01').timestamp()
tfe2 = pd.to_datetime('2018-06-01').timestamp()
vtfs2 = pd.to_datetime('2018-08-01').timestamp()
vtfe2 = pd.to_datetime('2018-10-01').timestamp()

id_col = 'fullVisitorId'
sort_col = 'visitStartTime'
tr_df1 = train_df_r.loc[(train_df_r[sort_col]>=tfs1) & (train_df_r[sort_col]<tfe1), [id_col,'target']].copy()
ts_df1 = train_df_r.loc[(train_df_r[sort_col]>=vtfs1) & (train_df_r[sort_col]<vtfe1), [id_col,'target']].copy()
print(f'tr_df1: {tr_df1.shape}, ts_df1: {ts_df1.shape}')
tr_df2 = train_df_r.loc[(train_df_r[sort_col]>=tfs2) & (train_df_r[sort_col]<tfe2), [id_col,'target']].copy()
ts_df2 = train_df_r.loc[(train_df_r[sort_col]>=vtfs2) & (train_df_r[sort_col]<vtfe2), [id_col,'target']].copy()
del train_df_r
gc.collect()
print(f'tr_df2: {tr_df2.shape}, ts_df2: {ts_df2.shape}')
tr_df1 = tr_df1.groupby(id_col).sum()
ts_df1 = ts_df1.groupby(id_col).sum()
gc.collect()
print(f'tr_df1: {tr_df1.shape}, ts_df1: {ts_df1.shape}')
tr_df2 = tr_df2.groupby(id_col).sum()
ts_df2 = ts_df2.groupby(id_col).sum()
gc.collect()
print(f'tr_df2: {tr_df2.shape}, ts_df2: {ts_df2.shape}')
df1 = tr_df1.join(ts_df1, lsuffix='_ls')
df2 = tr_df2.join(ts_df2, lsuffix='_ls')
print(f'df1: {df1.shape}, df2: {df2.shape}')
print(f"df1's target not null: {np.sum(df1.target.notnull())}, df2's target not null: {np.sum(df2.target.notnull())}")
df1 = np.log1p(df1.fillna(0))
df2 = np.log1p(df2.fillna(0))
gc.collect()
print(f'df1.target>0: {np.sum(df1.target>0)}, df2.target>0: {np.sum(df2.target>0)}')
print(np.intersect1d(df1.index, df2.index).shape)
cids = np.intersect1d(df1.loc[df1.target>0].index, df2.loc[df2.target>0].index)
print(cids.shape)
df1.loc[cids]
df2.loc[cids]
s1 = df1.target
print(s1.describe())
print()
s1 = s1.loc[s1>0]
print(s1.describe())
print('----------------------------------')
s2 = df2.target
print(s2.describe())
print()
s2 = s2.loc[s2>0]
print(s2.describe())
s1 = df1.target_ls
print(s1.describe())
print()
s1 = s1.loc[s1>0]
print(s1.describe())
print('----------------------------------')
s2 = df2.target_ls
print(s2.describe())
print()
s2 = s2.loc[s2>0]
print(s2.describe())
s1 = df1.loc[(df1.target_ls>0)&(df1.target>0), 'target']
print(s1.describe())
print('----------------------------------')
s2 = df2.loc[(df2.target_ls>0)&(df2.target>0), 'target']
print(s2.describe())
s1 = df1.loc[(df1.target_ls>0)&(df1.target>0), 'target_ls']
print(s1.describe())
print('----------------------------------')
s2 = df2.loc[(df2.target_ls>0)&(df2.target>0), 'target_ls']
print(s2.describe())
s1 = df1.target_ls - df1.target
print(s1.describe())
print()
s1 = s1.loc[s1!=0]
print(s1.describe())
print('----------------------------------')
s2 = df2.target_ls -df2.target
print(s2.describe())
print()
s2 = s2.loc[s2!=0]
print(s2.describe())
s1 = df1.loc[(df1.target_ls>0)&(df1.target>0), 'target_ls'] - df1.loc[(df1.target_ls>0)&(df1.target>0), 'target']
print(s1.describe())
print()
s1 = s1.loc[s1!=0]
print(s1.describe())
print('----------------------------------')
s2 = df2.loc[(df2.target_ls>0)&(df2.target>0), 'target_ls'] - df2.loc[(df2.target_ls>0)&(df2.target>0), 'target']
print(s2.describe())
print()
s2 = s2.loc[s2!=0]
print(s2.describe())
del tr_df1,ts_df1,tr_df2,ts_df2,df1,df2
gc.collect()
train_df_w = pd.read_pickle(os.path.join(data_dir, 'train_df_w'), compression='gzip')
test_df_w = pd.read_pickle(os.path.join(data_dir, 'test_df_w'), compression='gzip')
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')
train_df_u = pd.read_pickle(os.path.join(data_dir, 'train_df_u'), compression='gzip')
test_df_u = pd.read_pickle(os.path.join(data_dir, 'test_df_u'), compression='gzip')
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
train_df = train_df_w.join(train_df_u)
test_df = test_df_w.join(test_df_u)
del train_df_w,test_df_w,train_df_u,test_df_u
gc.collect()
print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
print(np.sum(train_df.isnull().values))
print(np.sum(test_df.isnull().values))
sort_col = 'visitStartTime_w_min'
train_df = train_df.reset_index()
gc.collect()
train_df = train_df.sort_values(by=[id_col, sort_col])
gc.collect()
train_df = train_df.reset_index(drop=True)
gc.collect()

test_df = test_df.reset_index()
gc.collect()
test_df = test_df.sort_values(by=[id_col, sort_col])
gc.collect()
test_df = test_df.reset_index(drop=True)
gc.collect()
def label_target(tfs, tfe, vtfs, vtfe):
    tr_df = train_df.loc[(((train_df[tm_col1]>=tfs) & (train_df[tm_col1]<tfe))
                         | ((train_df[tm_col2]>=tfs) & (train_df[tm_col2]<tfe)))]
    vids = train_df.loc[(((train_df[tm_col1]>=vtfs) & (train_df[tm_col1]<vtfe))
                         | ((train_df[tm_col2]>=vtfs) & (train_df[tm_col2]<vtfe))), id_col].unique()
    tr_df = tr_df.loc[~tr_df[id_col].isin(vids)]
    gc.collect()
    tr_df = tr_df.loc[tr_df.groupby(id_col)[tm_col1].idxmax()].copy()
    tr_df['target'] = 0
    
    return tr_df
tm_col1 = 'visitStartTime_w_min'
tm_col2 = 'visitStartTime_w_max'
tr_df1 = label_target(tfs1, tfe1, vtfs1, vtfe1)
gc.collect()
tr_df2 = label_target(tfs2, tfe2, vtfs2, vtfe2)
gc.collect()
print(f'tr_df1: {tr_df1.shape}, tr_df2: {tr_df2.shape}')
ids3 = train_df.loc[(((train_df[tm_col1]>=tfe1) & (train_df[tm_col2]<tfs2)) | (train_df[tm_col2]<tfs1)), id_col].unique()
tr_df3 = train_df.loc[train_df[id_col].isin(ids3)]
tr_df3 = tr_df3.loc[tr_df3.groupby(id_col)[tm_col1].idxmax()]
tr_df3 = tr_df3.loc[(((tr_df3[tm_col1]>=tfe1) & (tr_df3[tm_col2]<tfs2)) | (tr_df3[tm_col2]<tfs1))].copy()
gc.collect()
tr_df3['target'] = 0
print(f'tr_df3: {tr_df3.shape}')
cnt = train_df[id_col].value_counts()
ids4 = cnt.loc[cnt>1].index.values
tr_df4 = train_df.loc[train_df[id_col].isin(ids4)].copy()
gc.collect()
print(f'tr_df4: {tr_df4.shape}')
t_col = 'target_w_sum'
tr_df4['target'] = -1
gp = tr_df4.groupby(id_col)
cols = []
for i in range(2, 5):
    col = f'{t_col}_{i}'
    cols.append(col)
    tr_df4[col] = gp[t_col].shift(-i)
    col = f'{tm_col1}_{i}'
    cols.append(col)
    tr_df4[col] = gp[tm_col1].shift(-i)
    col = f'{tm_col2}_{i}'
    cols.append(col)
    tr_df4[col] = gp[tm_col2].shift(-i)
for i in range(2, 5):
    ind4 = (tr_df4['target'] < 0) & tr_df4[f'{t_col}_{i}'].notnull() & (
        tr_df4[f'{tm_col2}_{i}'] - tr_df4[tm_col1] > 60 * 24 * 3600)
    ind40 = ind4 & (tr_df4[f'{tm_col1}_{i}'] - tr_df4[tm_col2] > 9 * 30 * 24 * 3600)
    tr_df4.loc[ind40, 'target'] = 0
    ind41 = ind4 & (tr_df4[f'{tm_col1}_{i}'] - tr_df4[tm_col2] <= 9 * 30 * 24 * 3600)
    tr_df4.loc[ind41, 'target'] = tr_df4.loc[ind41, f'{t_col}_{i}']
tr_df4 = tr_df4.loc[tr_df4.target>=0].drop(cols, axis=1)
gc.collect()
print(f'tr_df4: {tr_df4.shape}')
train_df = tr_df1.append([tr_df2,tr_df3,tr_df4], sort=False)
gc.collect()
train_df = train_df.sort_values(by=[id_col, sort_col])
gc.collect()
train_df = train_df.reset_index(drop=True)
gc.collect()

train_df = train_df.drop('target_w_sum', axis=1)
gc.collect()
test_df = test_df.drop('target_w_sum', axis=1)
gc.collect()
print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
train_df = train_df.drop('window_id', axis=1)
test_df = test_df.loc[0==test_df.window_id].drop('window_id', axis=1)
print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
sub_df = pd.read_csv('../input/ga-customer-revenue-prediction/sample_submission_v2.csv', dtype={id_col: 'str'})
print(sub_df.shape)
s = test_df[id_col].unique()
sub_ids = sub_df[id_col].unique()
print(s.shape, sub_ids.shape)
sub_ids = np.intersect1d(s, sub_ids)
del sub_df, s
gc.collect()
print(sub_ids.shape)
def find_still_cols(df, diff_num_threshold=30):
    still_cols1,still_cols2 = [],[]
    for col in df.columns:
        cnt = df[col].value_counts(dropna=False)
        if cnt.shape[0] <= 1:
            still_cols1.append(col)
        elif df.shape[0] - cnt.iloc[0] < diff_num_threshold:
            still_cols2.append(col)
    return still_cols1,still_cols2
still_cols1,still_cols2 = find_still_cols(train_df)
print(f'still_cols1({len(still_cols1)}): {still_cols1}')
print(f'still_cols2({len(still_cols2)}): {still_cols2}')
still_cols = still_cols1 + still_cols2
train_df = train_df.drop(still_cols, axis=1)
gc.collect()
test_df = test_df.drop(still_cols, axis=1)
gc.collect()
print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
def find_same_cols(df, diff_num_threshold=30, detail=False):
    same_cols = set()
    for col1 in df.columns:
        if col1 not in same_cols:
            column1 = df[col1]
            for col2 in df.columns:
                if col2 > col1 and col2 not in same_cols:
                    diff_num = np.sum(column1 != df[col2])
                    if diff_num < diff_num_threshold:
                        if detail:
                            print(f'{col1} - {col2} = {diff_num}')
                        same_cols.add(col2)
    return list(same_cols)
same_cols = sorted(find_same_cols(train_df, detail=True))
print(f'same_cols({len(same_cols)}): {same_cols}')
train_df = train_df.drop(same_cols, axis=1)
gc.collect()
test_df = test_df.drop(same_cols, axis=1)
gc.collect()
print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')
cols = [col for col in test_df.columns if col[-1].isdigit()]
for col in tqdm(cols):
    cnt = Counter(train_df[col])
    cnt_pair = sorted(cnt.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
    unseen_value = -127 if -127 in cnt else '-127' if '-127' in cnt else cnt_pair[0][0]
    data_type = test_df[col].dtype
    test_df[col] = test_df[col].apply(lambda ele: unseen_value if ele not in cnt else ele).astype(data_type)
    del cnt,cnt_pair
    gc.collect()
train_df
test_df
y = train_df.target
print(y.describe())
print()
print(y.loc[y>0].describe())
s = test_df.dtypes
ss = list(s.loc[s == np.object].index)
print(len(ss))
print(ss)
ss = list(s.loc[s != np.object].index)
print(len(ss))
print(ss)
df = train_df
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
df = test_df
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
