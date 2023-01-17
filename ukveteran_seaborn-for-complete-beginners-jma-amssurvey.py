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



dat = pd.read_csv('../input/american-math-society-survey-data/AMSsurvey.csv')
dat.info()
dat['count'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count11']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count11']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count']

x = pd.Series(x, name="variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count11']

x = pd.Series(x, name="variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['count11']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
g = sns.JointGrid(x="count", y="count11", data=dat, space=0)

g = g.plot_joint(sns.kdeplot, cmap="Blues_d")

g = g.plot_marginals(sns.kdeplot, shade=True)