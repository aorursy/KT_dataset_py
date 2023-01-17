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
dat = pd.read_csv("../input/cd4-counts-for-hivpositive-patients/cd4.csv")
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
sns.regplot(x=dat['baseline'], y=dat['oneyear'])
plt.style.use('fast')

sns.jointplot(x='baseline', y='oneyear', data=dat)

plt.show()
sns.lineplot(x='baseline', y='oneyear', data=dat)