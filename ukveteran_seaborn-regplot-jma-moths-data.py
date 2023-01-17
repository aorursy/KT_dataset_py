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



dat = pd.read_csv('../input//moths-data/moths.csv')
dat
import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x="A", y="P", data=dat)
ax = sns.regplot(x="A", y="P", color="g", data=dat)
x = sns.regplot(x="A", y="P", marker="+", data=dat)

ax = sns.regplot(x="A", y="P", ci=68, data=dat)

ax = sns.regplot(x="A", y="P",x_jitter=.1, data=dat)

ax = sns.regplot(x="A", y="P", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="A", y="P",x_bins=4, data=dat)

ax = sns.regplot(x="A", y="P",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="A", y="P",  logistic=True, n_boot=500, data=dat)
x = sns.regplot(x="A", y="P", marker="+", data=dat)
ax = sns.regplot(x="A", y="P", ci=68, data=dat)
ax = sns.regplot(x="A", y="P",x_jitter=.1, data=dat)
ax = sns.regplot(x="A", y="P", x_estimator=np.mean, data=dat)
ax = sns.regplot(x="A", y="P",x_bins=4, data=dat)
ax = sns.regplot(x="A", y="P",x_estimator=np.mean, logx=True, data=dat)
ax = sns.regplot(x="A", y="P",  logistic=True, n_boot=500, data=dat)
x = sns.regplot(x="meters", y="P", marker="+", data=dat)

ax = sns.regplot(x="meters", y="P", ci=68, data=dat)

ax = sns.regplot(x="meters", y="P",x_jitter=.1, data=dat)

ax = sns.regplot(x="meters", y="P", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="meters", y="P",x_bins=4, data=dat)

ax = sns.regplot(x="meters", y="P",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="meters", y="P",  logistic=True, n_boot=500, data=dat)
x = sns.regplot(x="A", y="meters", marker="+", data=dat)

ax = sns.regplot(x="A", y="meters", ci=68, data=dat)

ax = sns.regplot(x="A", y="meters",x_jitter=.1, data=dat)

ax = sns.regplot(x="A", y="meters", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="A", y="meters",x_bins=4, data=dat)

ax = sns.regplot(x="A", y="meters",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="A", y="meters",  logistic=True, n_boot=500, data=dat)