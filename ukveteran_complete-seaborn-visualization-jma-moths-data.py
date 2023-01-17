import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.set(color_codes=True)
df = pd.read_csv('../input//moths-data/moths.csv')

df
x = df['meters']

sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.kdeplot(x, shade=True, cut=0)

sns.rugplot(x);
x = df['meters']

sns.distplot(x, kde=False, fit=stats.gamma);
sns.jointplot(x="meters", y="A", data=df);
sns.jointplot(x="A", y="P", data=df);
with sns.axes_style("white"):

    x=df['A']

    y=df['P']

    sns.jointplot(x=x, y=y, kind="hex", color="k")
sns.jointplot(x="A", y="P", data=df, kind="kde");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.A, df.P, ax=ax)

sns.rugplot(df.A, color="g", ax=ax)

sns.rugplot(df.P, vertical=True, ax=ax);
sns.pairplot(df)
g = sns.PairGrid(df)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6);