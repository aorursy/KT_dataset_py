from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/pulse-rates-and-exercise/Pulse.csv')

dat.head()
sns.set(rc={'figure.figsize':(19.7,8.27)})

sns.heatmap(dat.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(dat["Active"])
sns.scatterplot(x='Hgt',y='Wgt',data=dat)
sns.countplot(dat["Smoke"])
sns.countplot(dat["Sex"])
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = dat.groupby('Hgt').size()/dat['Hgt'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
plt.figure(figsize=(10,6))

sns.catplot(x="Hgt", y="Wgt", data=dat);

plt.ioff()