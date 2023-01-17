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



from sklearn.neighbors import NearestNeighbors

from numba import jit
# Read in features from GitHub

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')



print('Training data shape: ', train_data.shape)

print('Testing data shape:  ', test_data.shape)
train_data.describe()
%%time

idx = features = train_data.columns.values[2:202]

for df in [test_data, train_data]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
train_data[train_data.columns[202:]].head()
%%time

X = train_data.drop(['ID_code', 'target'], axis=1)

X_test = test_data.drop(['ID_code'], axis=1)

neigh = NearestNeighbors(4, n_jobs=-1)

neigh.fit(X)
%%time



dists, _ = neigh.kneighbors(X, n_neighbors=4)

mean_dist = dists.mean(axis=1)

max_dist = dists.max(axis=1)

min_dist = dists.min(axis=1)



train_data['mean_dist_4'] = mean_dist

train_data['max_dist_4'] = max_dist

train_data['min_dist_4'] = min_dist
%%time



test_dists, _ = neigh.kneighbors(X_test, n_neighbors=3)

test_mean_dist = test_dists.mean(axis=1)

test_max_dist = test_dists.max(axis=1)

test_min_dist = test_dists.min(axis=1)



test_data['mean_dist_4'] = test_mean_dist

test_data['max_dist_4'] = test_max_dist

test_data['min_dist_4'] = test_min_dist
train_data[train_data.columns[210:]].head()
train_data.to_csv('train_milos4.csv', index=False)

test_data.to_csv('test_milos4.csv', index=False)

sample_submission.to_csv('sample_submission.csv', index=False)

print('Training data shape: ', train_data.shape)

print('Testing data shape:  ', test_data.shape)