# import modules

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)



import lightgbm as lgb

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MultiLabelBinarizer

import collections



import matplotlib.pyplot as plt

import seaborn as sns

import re, time



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')

plt.style.use('seaborn')

sns.set(font_scale=1)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pwd
# data load

time_begin = time.time()

df_train = pd.read_csv('../input/expedia-personalized-sort/data/train.csv')

df_test = pd.read_csv('../input/expedia-personalized-sort/data/test.csv')

time_end = time.time()

print('read csv time:' + str(time_end - time_begin))
df_train.head(5)
df_test.head(5)