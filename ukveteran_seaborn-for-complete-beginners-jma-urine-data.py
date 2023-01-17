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



dat = pd.read_csv('../input/urine-analysis-data/urine.csv')
dat.info()
dat['r'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

x = pd.Series(x, name="Urea")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

x = pd.Series(x, name="Calc")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

x = pd.Series(x, name="Urea")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

x = pd.Series(x, name="Calc")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

x = pd.Series(x, name="Urea")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

x = pd.Series(x, name="Calc")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['urea']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = dat['calc']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
dat['urea'].nunique()
dat['calc'].nunique()
dat['urea'].value_counts()
dat['calc'].value_counts()
g = sns.catplot(x="r", kind="count", palette="ch:.25", data=dat)
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="urea", y="calc", hue="r", 

                   data=dat, palette="Set2", size=20, marker="D",

                   edgecolor="gray", alpha=.25)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["urea"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=dat["calc"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="urea", y="calc", data=dat)

plt.show()
datt =dat[['r','urea', 'calc']]

g = sns.PairGrid(datt)

g = g.map(plt.scatter)
g = sns.PairGrid(datt)

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
g = sns.PairGrid(datt, hue="r")

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
g = sns.PairGrid(datt, hue="r")

g = g.map_diag(plt.hist, histtype="step", linewidth=3)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
g = sns.PairGrid(datt, vars=['urea', 'calc'])

g = g.map(plt.scatter)
g = sns.PairGrid(datt)

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.map_diag(sns.kdeplot, lw=3, legend=False)
g = sns.JointGrid(x="urea", y="calc", data=dat)

g = g.plot(sns.regplot, sns.distplot)