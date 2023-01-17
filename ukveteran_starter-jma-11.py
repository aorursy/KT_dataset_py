import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

import matplotlib.ticker as mtick

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

plt.xkcd() 
cycle = pd.read_csv("../input/cycling.csv")

cycle.head()
corr_mat = cycle.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.matshow(cycle.corr())

plt.colorbar()

plt.show()
sns.boxplot(cycle['Distance'])
p = cycle.hist(figsize = (20,20))
dcycle = pd.read_csv("../input/Data Cycling.csv")

dcycle.head()
corr_mat = dcycle.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.matshow(dcycle.corr())

plt.colorbar()

plt.show()
q = dcycle.hist(figsize = (20,20))