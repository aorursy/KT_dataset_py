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



dat = pd.read_csv('../input/sparrow-measurements/Sparrows.csv')
dat.head
dat.info()
dat['WingLength'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

x = pd.Series(x, name="Wing Length variable")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

x = pd.Series(x, name="Wing Length variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

x = pd.Series(x, name="Wing Length variable")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['WingLength']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
dat['Weight'].nunique()
dat['Weight'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Weight", data=dat, color="c")

plt.show()
g = sns.catplot(x="Weight", kind="count", palette="ch:.25", data=dat)
g = sns.catplot(x="WingLength", kind="count", palette="ch:.25", data=dat)
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="Weight", y="WingLength", hue="WingLength", 

                   data=dat, palette="Set2", size=20, marker="D",

                   edgecolor="gray", alpha=.25)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="Weight", y="WingLength", hue="Weight", 

                   data=dat, palette="Set2", size=20, marker="D",

                   edgecolor="gray", alpha=.25)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["Weight"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["WingLength"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="Weight", y="WingLength", data=dat)

plt.show()
datt =dat[['Weight', 'WingLength']]

g = sns.PairGrid(datt)

g = g.map(plt.scatter)
g = sns.PairGrid(datt)

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
g = sns.PairGrid(datt, hue="Weight")

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
g = sns.PairGrid(datt, hue="Weight")

g = g.map_diag(plt.hist, histtype="step", linewidth=3)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
g = sns.PairGrid(datt, vars=['Weight', 'WingLength'])

g = g.map(plt.scatter)
g = sns.PairGrid(datt)

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.map_diag(sns.kdeplot, lw=3, legend=False)
g = sns.JointGrid(x="Weight", y="WingLength", data=dat)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="Weight", y="WingLength", data=dat, space=0)

g = g.plot_joint(sns.kdeplot, cmap="Blues_d")

g = g.plot_marginals(sns.kdeplot, shade=True)
g = sns.JointGrid(x="Weight", y="WingLength", data=dat, height=5, ratio=2)

g = g.plot_joint(sns.kdeplot, cmap="Reds_d")

g = g.plot_marginals(sns.kdeplot, color="r", shade=True)