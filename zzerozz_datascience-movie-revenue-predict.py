# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import resource
from os import walk
import keras
from datetime  import datetime,timedelta,date
import re
import gc
import matplotlib.pyplot as plt
import pickle

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def save_obj(obj, file_path):    
    file_save = open(file_path, 'wb')
    pickle.dump(obj, file_save, pickle.HIGHEST_PROTOCOL)
    file_save.close()
    
def load_obj(path):
    file_save = open(path, 'rb')
    return pickle.load(file_save)
df = load_obj('../input/tmp_df_5_1')
df_info = load_obj('../input/tmp_dfinfo_5_1')
df.drop_duplicates(subset='id', inplace=True)
df.head()
try: 
    df.release_date = df.release_date.astype('str')
except:
    print('-')
df_tmp = df[['id', 'people']]
tmp_all = []

for value in df_tmp.values:
    tmp = []
    for person in value[1]:
        try:
            tmp.append(df_info[(df_info.movie == value[0]) & (df_info.person_name == person)].person_feats.values[0])
        except:
            tmp.append(None)
    tmp_all.append(tmp)
tmp_person_feats = []
for temp_val in tmp_all:
    try:
        tmp_person_feats.append(np.concatenate([np.min(temp_val, axis=0), np.max(temp_val, axis=0),\
             np.mean(temp_val, axis=0), np.std(temp_val, axis=0), np.median(temp_val, axis=0)]).tolist())
    except:
        tmp_person_feats.append([-1]*42*5)
df_people_feats = pd.DataFrame(tmp_person_feats)
df_people_feats.columns = ['_c' + str(i) for i in range(42*5)]
df_people_feats['id'] = df.id
# df_people_feats['people'] was deleted
df = pd.merge(df, df_people_feats.drop_duplicates(subset='id'), on='id', how='left')
df.head()
df_studio_feats = pd.DataFrame(df.studio_feats.values.tolist())
df_studio_feats.columns = ['_c' + str(i) for i in range(210, 252)]
df_studio_feats['id'] = df.id
df = pd.merge(df, df_studio_feats, on='id', how='left')
df.fillna(-1, inplace=True)
df = df[~df.lastseen.str.contains('2018')]
df.drop(['studio_feats'], axis=1, inplace=True)
df.head()
from sklearn.preprocessing import  OneHotEncoder, StandardScaler

df['year'] = df.release_date.apply(lambda x: int(x[:4]))

for feat in ['mpaa_rating', 'genre', 'season', 'movie_season']:
    onehot = OneHotEncoder()
    a = onehot.fit_transform(df[feat].values.reshape(-1,1))
    df_feat =pd.DataFrame(a.toarray())
    df_feat.columns = [feat + '_' + str(i) for i in range(df[feat].unique().shape[0])]
    
    df_feat['id'] = df.id
    df = pd.merge(df, df_feat, on='id', how='left')
    
a = keras.utils.to_categorical(df['month'].values, dtype=np.int8)
df_month = pd.DataFrame(a[:, 1:])

df_month.columns = ['month' + '_' + str(i) for i in range(12)]
df_month['id'] = df.id
df = pd.merge(df, df_month, on='id', how='left')

del a, df_month
gc.collect()
drop_feats = ['people', 'studio', 'genre', 'id', 'lastseen', 'mpaa_rating',\
    'name', 'release_date', 'revenue', 'month', 'season', 'movie_season', 'year']

df.head()
df.to_csv('traintest.csv', index=False)

del tmp_person_feats, tmp, tmp_all, df_info, df_tmp

gc.collect()
df_train = df[df.year < 2016]
df_test = df[df.year >= 2016]

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

df_train.shape, df_test.shape
df.release_date.values[0]

