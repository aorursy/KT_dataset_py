# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns 
import itertools
import warnings
warnings.filterwarnings("ignore")
Train_df =  pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
test_df = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")
Train_df
Train_df.info() 
print("TRAIN ",Train_df.memory_usage().sum()/1024**2)
print("TEST ",test_df.memory_usage().sum()/1024**2)
"""
Check Data Type 
"""
col = Train_df['assists'].dtype
type(col)
'''
All type of min and max value in INT & floar
'''
print('INT8 ',np.iinfo(np.int8))
print('INT16 ',np.iinfo(np.int16))
print('INT32 ',np.iinfo(np.int32))
print('INT64 ',np.iinfo(np.int64))
print('FLOAT32 ',np.finfo(np.float32))
print('FLOAT64 ',np.finfo(np.float64))
"""
Reduce Memory function
"""
def reduce_memory(df):
    total_memoryIN_mebi = df.memory_usage().sum()/1024**2 # Convert Bytes to Mebibyte
    
    for col in df.columns: # get column one by one
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()  # min value of column
            c_max = df[col].max() # max value of column
            
            if str(col_type)[:3] == 'int': # convert numpy.dtype to string
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    after_reduce = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        total_memoryIN_mebi, after_reduce, 100 * (total_memoryIN_mebi - after_reduce) / total_memoryIN_mebi))
    return df

%%time

train = reduce_memory(Train_df)
test = reduce_memory(test_df)
print(train.shape,test.shape)
train.info()
train.columns
for i in ['Id','groupId','matchId']:  # Name in List formate
    print(f'unique [{i}] count:', train[i].nunique()) #Getting Unique Data from Data Sets
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fig, ax = plt.subplots(1, 2, figsize=(30,6))
train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train['matchType'] = train['matchType'].apply(mapper)
train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])
def mergeList(list1,list2):
    return list(itertools.product(list1,list2))
match = train.groupby(['matchType','matchId']).size().to_frame('players in match')
group = train.groupby(['matchType','matchId','groupId']).size().to_frame('players in group')
pd.concat([match.groupby('matchType').describe()[mergeList(['players in match'],['min','mean','max'])], 
           group.groupby('matchType').describe()[mergeList(['players in group'],['min','mean','max'])]], axis=1)
print(group['players in group'].nlargest())
''' ex) matchId=='3e029737889ce9', groupId=='b8275198faa03b'
'''
subset = train[train['matchId']=='3e029737889ce9']
sub_grp = subset[subset['groupId']=='b8275198faa03b']

print('matchId ==\'3e029737889ce9\' & groupId ==\'b8275198faa03b\'')
print('-'*50)
print('players:',len(subset))
print('groups:',subset['groupId'].nunique())
print('numGroups:',subset['numGroups'].unique())
print('maxPlace:',subset['maxPlace'].unique())
print('-'*50)
print('max-group players:',len(sub_grp))
print('max-group winPlacePerc:',sub_grp['winPlacePerc'].unique())
print('-'*50)
print('winPlacePerc:',subset['winPlacePerc'].sort_values().unique())
corr = train.corr()
f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(train.corr(), annot=True, fmt= '.1f',ax=ax, cmap="BrBG")
sns.set(font_scale=1.25)
plt.show()
