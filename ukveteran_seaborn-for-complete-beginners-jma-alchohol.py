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



dat = pd.read_csv('../input/alcohol-solubility-in-water-data/alcohol.csv')
dat.head
dat.info()
dat['logPC'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['logPC']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Mass']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['logPC']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Mass']

x = pd.Series(x, name="variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Mass']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="logPC", y="Mass", data=dat)

plt.show()
g = sns.JointGrid(x="logPC", y="Mass", data=dat, space=0)

g = g.plot_joint(sns.kdeplot, cmap="Blues_d")

g = g.plot_marginals(sns.kdeplot, shade=True)
g = sns.JointGrid(x="logPC", y="Mass", data=dat, height=5, ratio=2)

g = g.plot_joint(sns.kdeplot, cmap="Reds_d")

g = g.plot_marginals(sns.kdeplot, color="r", shade=True)
g = sns.JointGrid(x="logPC", y="Mass", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.PairGrid(dat, vars=['logPC', 'Mass'])

g = g.map(plt.scatter)