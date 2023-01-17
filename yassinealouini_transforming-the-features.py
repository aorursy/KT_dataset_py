import pandas as pd

import numpy as np

from scipy.stats import norm

import seaborn as sns

import matplotlib.pylab as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df['var_0'].plot(kind='hist', bins=100)
var_0_mean = train_df['var_0'].mean()

var_0_std = train_df['var_0'].std()
var_0_mean, var_0_std
scaled_var_0 = (train_df['var_0'] - var_0_mean) / var_0_std

trf_var_0 = pd.Series(norm.cdf(scaled_var_0))
scaled_var_0.plot(kind='hist', bins=100)
trf_var_0.plot(kind='hist', bins=100)
def uniformize(s):

    mean = s.mean()

    std = s.std()

    return pd.Series(norm.cdf((s - mean) / std))
uniformize(train_df['var_1']).plot(kind='hist', bins=100)
for col in train_df.drop(['target', 'ID_code'], axis=1).columns[:10]:

    _, ax = plt.subplots(1, 1)

    uniformize(train_df.loc[lambda df: df.target == 0, col]).plot(kind='hist', 

                                                                  bins=100, ax=ax,

                                                                 label="0")

    uniformize(train_df.loc[lambda df: df.target == 1, col]).plot(kind='hist', 

                                                                  bins=100, ax=ax,

                                                                 label="1")

    ax.set_title(f'Normal transformation for {col}')

    ax.legend()