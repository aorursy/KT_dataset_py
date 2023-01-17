import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc

import json
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import trange,tqdm
from collections import Counter

pd.options.display.max_rows = 20
pd.options.display.max_columns = 999
print(os.listdir("../input"))
train_df = pd.read_pickle('../input/gs-get-src-data/train_df', compression='gzip')
print(train_df.shape)
test_df = pd.read_pickle('../input/gs-get-src-data/test_df', compression='gzip')
print(test_df.shape)
train_df_u = pd.read_pickle('../input/gs-get-src-data/train_df_u', compression='gzip')
print(train_df_u.shape)
test_df_u = pd.read_pickle('../input/gs-get-src-data/test_df_u', compression='gzip')
print(test_df_u.shape)
id_col = 'fullVisitorId'
cnt = train_df[id_col].value_counts()
ids = cnt.loc[cnt>1].index.values
print(ids.shape)
df = train_df.loc[train_df[id_col].isin(ids)]
print(df.shape)
gdf = df.groupby(id_col)['target'].sum()
tids = gdf.loc[gdf>0].index.values
print(tids.shape)
tdf = df.loc[df[id_col].isin(tids)]
print(tdf.shape)
gdf = tdf.loc[:,[id_col,'target']]
gdf['target'] = gdf['target']>0
gdf = gdf.groupby(id_col)['target'].sum()
tids2 = gdf.loc[gdf>1].index.values
print(tids2.shape)
tdf2 = tdf.loc[tdf[id_col].isin(tids2)]
print(tdf2.shape)
cols = list(tdf2.columns)
cols = [col for col in cols if col != id_col] + [id_col]
tdf2[cols].loc[tdf2[id_col]==tids2[3]]
tdf2.groupby([id_col,'visit_year'])['target'].sum().reset_index()
cids = np.intersect1d(train_df.fullVisitorId, test_df.fullVisitorId)
print(len(cids))
print(train_df.loc[train_df.fullVisitorId.isin(cids)].shape)
print(test_df.loc[test_df.fullVisitorId.isin(cids)].shape)
cdf = tdf.loc[tdf[id_col].isin(cids)].groupby(id_col)['target'].sum()
print(cdf.shape)
cdf
sub = pd.read_csv('../input/gs-lgb-uv-ensemble-1-1/lgb_uv_ensemble_1_1.csv')
print(sub.shape)
sub1 = sub.join(cdf, on=id_col)
ind = sub1.target.notnull()
sub1.loc[ind, 'PredictedLogRevenue'] = (sub1.loc[ind, 'PredictedLogRevenue']+np.log1p(sub1.loc[ind, 'target']))/2
submission = sub1.drop('target', axis=1)
submission
submission.to_csv('test_all.csv', index=False, float_format='%.5f')