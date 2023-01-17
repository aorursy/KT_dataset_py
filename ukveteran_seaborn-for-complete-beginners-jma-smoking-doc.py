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



dat = pd.read_csv('../input/smoking-deaths-among-doctors/breslow.csv')
dat.info()
dat['smoke'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['ns']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['y']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['n']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['ns']

x = pd.Series(x, name="NS")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['y']

x = pd.Series(x, name="Y")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['n']

x = pd.Series(x, name="N")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['ns']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['n']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['y']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['ns']

x = pd.Series(x, name="NS")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['n']

x = pd.Series(x, name="N")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['y']

x = pd.Series(x, name="Y")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['ns']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['n']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['y']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
g = sns.catplot(x="smoke", kind="count", palette="ch:.25", data=dat)
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="n", y="y", hue="smoke", 

                   data=dat, palette="Set2", size=20, marker="D",

                   edgecolor="gray", alpha=.25)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["ns"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["n"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["y"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="n", y="y", data=dat)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="ns", y="y", data=dat)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="ns", y="n", data=dat)

plt.show()
g = sns.JointGrid(x="n", y="y", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="ns", y="y", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="ns", y="n", data=dat)

g = g.plot(sns.regplot, sns.distplot)