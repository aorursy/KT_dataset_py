import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set(style="whitegrid")

from collections import Counter

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')



dat = pd.read_csv('../input/treatment-of-migraine-headaches/KosteckiDillon.csv')
dat
import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x="age", y="airq", data=dat)
ax = sns.regplot(x="age", y="airq", color="g", data=dat)
ax = sns.regplot(x="age", y="airq", marker="+", data=dat)
ax = sns.regplot(x="age", y="airq", ci=68, data=dat)
ax = sns.regplot(x="age", y="time",x_jitter=.1, data=dat)
ax = sns.regplot(x="age", y="time", x_estimator=np.mean, data=dat)
ax = sns.regplot(x="age", y="time",x_bins=4, data=dat)
ax = sns.regplot(x="age", y="time",x_estimator=np.mean, logx=True, data=dat)
ax = sns.regplot(x="age", y="time",  logistic=True, n_boot=500, data=dat)
ax = sns.regplot(x="airq", y="time",x_estimator=np.mean, logx=True, data=dat)
ax = sns.regplot(x="airq", y="time", x_estimator=np.mean, data=dat)
ax = sns.regplot(x="dos", y="time", x_estimator=np.mean, data=dat)
ax = sns.regplot(x="dos", y="time", x_estimator=np.mean, logx=True, data=dat)