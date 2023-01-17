# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
lines = open('../input/train.csv').readlines()
print(lines[1])
df_train = pd.read_csv('../input/train.csv')
df_train = df_train.replace(np.nan, -1)
df_train = df_train.replace('yes', 1)
df_train = df_train.replace('no', 0)
df_train = df_train.replace('yes', 1)
df_train = df_train.drop(df_train.columns[95],1)
df_train.head()
# print(df_train['v18q1'][0], np.nan)
train_x = np.array(df_train[df_train.columns[1:-1]])
print(','.join(df_train.columns[94:]))
train_x = train_x.astype(np.float32)
print(train_x.shape)

train_y = np.array(df_train['Target'])
print(train_y.shape)
print(set(train_y))
plt.figure(figsize=(20,10))
plt.hist(train_y)
plt.show()
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.15, random_state=22)
print(train_x.shape, test_x.shape)
import lightgbm as lgb
import datetime
train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(test_x, label=test_y, reference=train_data)
# lightgbm 
param = { 'num_leaves':31,'num_trees':100, 'objective':'multiclass', 'max_depth':50, 'learning_rate':.05, 'max_bin':200, 'num_class':5, 'is_unbalance':'true'}
param['metric'] = ['auc', 'acc']

# lightgbm
num_round = 500
# start = datetime.now()
lgbm = lgb.train(param,train_data,num_round, valid_sets=val_data, early_stopping_rounds=10)
lgbm.save_model('model.txt')
# load test set
df_test = pd.read_csv('../input/test.csv')
df_sample = pd.read_csv('../input/sample_submission.csv')
df_sample.head()
# 预处理
df_test = df_test.replace(np.nan, -1)
df_test = df_test.replace('yes', 1)
df_test = df_test.replace('no', 0)
df_test = df_test.replace('yes', 1)
df_test = df_test.drop(df_test.columns[95],1)
df_test.head()
test_x = df_test[df_test.columns[1:]].values
test_id = df_test['Id'].values
print(test_x.shape)
## Predict
predict = lgbm.predict(test_x)
print(np.argmax(predict,axis=1)[:10])
result = np.argmax(predict, axis=1)
submit = pd.DataFrame()
submit['Id'] = test_id
submit['Target'] = result
submit.head()
submit.to_csv('submit.csv', index=None)
