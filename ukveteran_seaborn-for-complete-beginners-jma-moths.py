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
dat.info()
dat['A'].value_counts()
dat['P'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['A']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['P']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['meters']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['A']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['P']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['meters']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['A']

x = pd.Series(x, name="A")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['P']

x = pd.Series(x, name="P")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['meters']

x = pd.Series(x, name="meters")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
g = sns.catplot(x="A", kind="count", palette="ch:.25", data=dat)
g = sns.catplot(x="P", kind="count", palette="ch:.25", data=dat)
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="A", y="P", data=dat)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="A", y="meters", data=dat)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="P", y="meters", data=dat)

plt.show()
g = sns.JointGrid(x="A", y="P", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="A", y="meters", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="P", y="meters", data=dat)

g = g.plot(sns.regplot, sns.distplot)