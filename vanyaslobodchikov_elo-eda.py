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
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_merchants = pd.read_csv('../input/merchants.csv')
df_merchants.head()
df_new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
df_new_merchant_transactions.head()
df_historical_transactions = pd.read_csv('../input/historical_transactions.csv')
df_historical_transactions.head()
dict_dfs = {'df_train': df_train, 
          'df_merchants': df_merchants, 
          'df_historical_transactions': df_historical_transactions, 
          'df_new_merchant_transactions': df_new_merchant_transactions}
df_train.describe()
df_train['target'].hist(bins=100)
# 2207 values are located on the left tail of the target distribution
sum((df_train['target'] > -33.3) & (df_train['target'] < -33))
# This left tail appears to be the minimum target value
max_target = np.max(df_train['target'])
min_target = np.min(df_train['target'])
left_tail_mask = (df_train['target'] == min_target)
print(sum(left_tail_mask))
min_to_9_mask = (df_train['target'] > min_target) & (df_train['target'] < -9)
sum(min_to_10_mask)
# Seems like the target distribution is normal except the left tail (minimum value)
df_train['target'][min_to_10_mask].hist(bins=10)
nine_to_max_mask = (df_train['target'] > 9) & (df_train['target'] <= max_target)
sum(nine_to_max_mask)
df_train['target'][nine_to_max_mask].hist(bins=10)
# Checking for missing values
def check_null(df):
    for col in df.columns:
        print('{}: {}'.format(col, sum(df[col].isnull())))
for df_name in dict_dfs:
    print('--------------Number of NaNs in {}----------------'.format(df_name))
    check_null(dict_dfs[df_name])
