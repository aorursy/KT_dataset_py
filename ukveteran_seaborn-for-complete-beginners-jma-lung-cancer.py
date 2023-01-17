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



dat = pd.read_csv('../input/lung-cancer-dataset/lung_cancer_examples.csv')
dat.info()
dat['Age'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Age']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Smokes']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Age']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Smokes']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Age']

x = pd.Series(x, name="variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Smokes']

x = pd.Series(x, name="variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Age']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['Smokes']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
g = sns.JointGrid(x="Age", y="Smokes", data=dat, space=0)

g = g.plot_joint(sns.kdeplot, cmap="Blues_d")

g = g.plot_marginals(sns.kdeplot, shade=True)