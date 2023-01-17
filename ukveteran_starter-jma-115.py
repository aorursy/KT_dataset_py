import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
dat = pd.read_csv("../input/pulse-rates-and-exercise/Pulse.csv")
dat.head()
df = pd.concat([dat], axis=1, join='inner').sort_index()

corr_mat = df.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
df = pd.concat([dat], axis=1, join='inner').sort_index()

plt.matshow(df.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.regplot(x=dat['Hgt'], y=dat['Wgt'])
plt.style.use('fast')

sns.jointplot(x='Hgt', y='Wgt', data=dat)

plt.show()
sns.lineplot(x='Hgt', y='Wgt', data=dat)