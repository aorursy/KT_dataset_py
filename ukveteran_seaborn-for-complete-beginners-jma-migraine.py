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
dat.info()
dat['age'].value_counts()
dat['airq'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['age']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['airq']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['age']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['airq']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['age']

x = pd.Series(x, name="Age")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['airq']

x = pd.Series(x, name="Airq")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
g = sns.catplot(x="age", kind="count", palette="ch:.25", data=dat)
g = sns.catplot(x="airq", kind="count", palette="ch:.25", data=dat)
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="age", y="airq", data=dat)

plt.show()
g = sns.JointGrid(x="age", y="airq", data=dat)

g = g.plot(sns.regplot, sns.distplot)