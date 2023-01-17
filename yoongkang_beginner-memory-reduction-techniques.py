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
train_identity = pd.read_csv('/kaggle/input/train_identity.csv')

train_transaction = pd.read_csv('/kaggle/input/train_transaction.csv')
print(train_identity.shape)

print(train_transaction.shape)
original_memory_usage = train_transaction.memory_usage()

original_memory_usage
total_original_memory_usage = original_memory_usage.sum()
a = (train_identity.dtypes == 'object')

id_cat_cols = list(a[a].index)

a = (train_transaction.dtypes == 'object')

trans_cat_cols = list(a[a].index)

id_num_cols = list(set(train_identity.columns) - set(id_cat_cols))

trans_num_cols = list(set(train_transaction.columns) - set(trans_cat_cols))
print(len(id_cat_cols))

print(len(id_num_cols))

print(len(trans_cat_cols))

print(len(trans_num_cols))
train_transaction.dtypes
train_transaction[trans_num_cols].isnull().sum().sort_values(ascending=False)
trans_integer_cols = []

for c in trans_num_cols:

    try:

        if train_transaction[c].fillna(-1.0).apply(float.is_integer).all():

            trans_integer_cols += [c]

    except Exception as e:

        print("error: ", c, e)

len(trans_integer_cols)
stats = train_transaction[trans_integer_cols].describe().transpose()

stats
int8columns = stats[stats['max'] < 256].index

print(int8columns.shape)

print(int8columns)
int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index

print(int16columns.shape)

print(int16columns)
int8columns.shape[0] + int16columns.shape[0]
for c in int8columns:

    train_transaction[f'{c}_isna'] = train_transaction[c].isnull()

    train_transaction[c].fillna(-1.0, inplace=True)

    train_transaction[c] = train_transaction[c].astype('int8')
for c in int16columns:

    train_transaction[f'{c}_isna'] = train_transaction[c].isnull()

    train_transaction[c].fillna(-1.0, inplace=True)

    train_transaction[c] = train_transaction[c].astype('int16')
new_memory_usage = train_transaction.memory_usage()

new_memory_usage
total_new_memory_usage = new_memory_usage.sum()

total_new_memory_usage
total_original_memory_usage
total_original_memory_usage - total_new_memory_usage
train_transaction.to_feather('train_transaction_reduced_memory')