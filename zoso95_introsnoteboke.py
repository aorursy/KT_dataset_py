# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
path = '../input/'

test = pd.read_csv(path + 'test_users.csv')
train = pd.read_csv(path + 'train_users_2.csv')
sessions = pd.read_csv(path + 'sessions.csv')
countries = pd.read_csv(path + 'countries.csv')


import collections
from scipy.cluster.vq import kmeans,vq
import tensorflow as tf
import hashlib
print(test.as_matrix()[1:2])
print(train.as_matrix()[1:2])
#filtered = train.query('country_destination == ["US", "NDF", "other"]')
#print(filtered.head())
def f(x):
    print(type(x))
    if type(x) == 'string':
        return hashlib.md5(x).hexdigest()
    return x
t = train.apply(f)
print(t.head())
#print(countries)
country_to_ind = {}
i = 0
for c in countries.iterrows():
    series = c[1]
    print(series)
    print(series.as_matrix())
    country_to_ind[series['country_destination']]=i
    i = i + 1
country_to_ind['NDF'] = i
country_to_ind['other'] = i + 1
print(country_to_ind)
   

train['country_destination'].apply(lambda x: country_to_ind[x]).hist()
print(train['country_destination'].apply(lambda x: country_to_ind[x]).describe())
def ind(x):
    if x == 10 or x == 11 or x==9:
        return 1
    return 0
train['country_destination'].apply(lambda x: country_to_ind[x]).apply(ind).sum() # 0.92 of total.

hist = collections.defaultdict(list)
for i in train.iterrows():
    series = i[1]
    key = ','.join([series[language],])
    hist[key].append(country_to_ind[series['country_destination']])
    print(series)
    break
print(hist)