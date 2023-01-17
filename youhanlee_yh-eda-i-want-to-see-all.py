import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

import warnings

import gc

import time

import sys

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')

from sklearn import metrics



plt.style.use('seaborn')

sns.set(font_scale=2)

pd.set_option('display.max_columns', 500)
DEBUG = False



if DEBUG:

    NROWS = 100000

else:

    NROWS = None
%%time

train = pd.read_csv('../input/train.csv', nrows=NROWS)
train['target'].value_counts().plot.bar()
%%time

# checking missing data

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data
for col in train.columns[2:]:

    print("Number of unique values of {} : {}".format(col, train[col].nunique()))
train['var_68'].value_counts()
corr = train.corr()
abs(corr['target']).sort_values(ascending=False)
target_mask = train['target'] == 1

non_target_mask = train['target'] == 0 
from scipy.stats import ks_2samp
statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])
statistics_array = []

for col in train.columns[2:]:

    statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])

    statistics_array.append(statistic)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.kdeplot(train.loc[non_target_mask, col], ax=ax, label='Target == 0')

    sns.kdeplot(train.loc[target_mask, col], ax=ax, label='Target == 1')



    ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))

    plt.show()