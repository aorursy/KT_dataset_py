import os

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math as mt

import scipy



import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px





from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadskawasakicsv/kawasaki.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'kawasaki.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Reds')

plt.show()
fig = go.Figure(data=[go.Scatter(

    x=df['Samples'][0:10],

    y=df['204252_at'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Kawasaki disease',

    xaxis_title="Samples",

    yaxis_title="204252_at",

)

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Samples').size()/df['204252_at'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
sns.pairplot(df);
df.corr()
sns.heatmap(df[df.columns[:]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()