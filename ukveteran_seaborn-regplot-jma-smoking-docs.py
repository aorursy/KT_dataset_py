import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

dat=pd.read_csv("../input/smoking-deaths-among-doctors/breslow.csv")

dat
x = sns.regplot(x="age", y="n", marker="+", data=dat)

ax = sns.regplot(x="age", y="n", ci=68, data=dat)

ax = sns.regplot(x="age", y="n",x_jitter=.1, data=dat)

ax = sns.regplot(x="age", y="n", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="age", y="n",x_bins=4, data=dat)

ax = sns.regplot(x="age", y="n",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="age", y="n",  logistic=True, n_boot=500, data=dat)
x = sns.regplot(x="age", y="y", marker="+", data=dat)

ax = sns.regplot(x="age", y="y", ci=68, data=dat)

ax = sns.regplot(x="age", y="y",x_jitter=.1, data=dat)

ax = sns.regplot(x="age", y="y", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="age", y="y",x_bins=4, data=dat)

ax = sns.regplot(x="age", y="y",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="age", y="y",  logistic=True, n_boot=500, data=dat)
x = sns.regplot(x="y", y="n", marker="+", data=dat)

ax = sns.regplot(x="y", y="n", ci=68, data=dat)

ax = sns.regplot(x="y", y="n",x_jitter=.1, data=dat)

ax = sns.regplot(x="y", y="n", x_estimator=np.mean, data=dat)

ax = sns.regplot(x="y", y="n",x_bins=4, data=dat)

ax = sns.regplot(x="y", y="n",x_estimator=np.mean, logx=True, data=dat)

ax = sns.regplot(x="y", y="n",  logistic=True, n_boot=500, data=dat)
ax1 = sns.regplot(x="y", y="n",x_bins=4, data=dat)
ax2 = sns.regplot(x="age", y="n",x_bins=4, data=dat)
ax3 = sns.regplot(x="y", y="age",x_bins=4, data=dat)