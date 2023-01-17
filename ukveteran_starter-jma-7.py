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
EMT = pd.read_csv("../input/pmsm_temperature_data.csv")

EMT.head()
EMT.describe()
corr_mat = EMT.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.matshow(EMT.corr())

plt.colorbar()

plt.show()
sns.boxplot(EMT['coolant'])
plt.scatter(EMT['motor_speed'], EMT['torque'])
p = EMT.hist(figsize = (20,20))
ax = sns.violinplot(x=EMT["coolant"])
ax = sns.violinplot(x=EMT["motor_speed"])
ax = sns.violinplot(x=EMT["torque"])
sns.distplot(EMT["coolant"])
sns.distplot(EMT["torque"])