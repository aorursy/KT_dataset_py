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
dat= pd.read_csv('../input/metadata.csv')

dat.head()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="10m_critical_power_hr",y='10s_critical_power',data=dat)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="20m_critical_power_hr",y='20s_critical_power',data=dat)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="30m_critical_power_hr",y='30s_critical_power',data=dat)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="10m_critical_power_hr",y='20m_critical_power_hr',data=dat)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="10m_critical_power_hr",y='30m_critical_power_hr',data=dat)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="20m_critical_power_hr",y='30m_critical_power_hr',data=dat)

plt.show()