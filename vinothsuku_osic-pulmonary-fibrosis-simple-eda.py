import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly as plty

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import iplot

from plotly.subplots import make_subplots

import plotly.io as pio

import os

%matplotlib inline

path = '../input/osic-pulmonary-fibrosis-progression/'
df_train = pd.read_csv(f'{path}train.csv')

df_test = pd.read_csv(f'{path}test.csv')
df_train.info()
df_train.describe(include='all').T
df_train.head()
df_tmp = df_train.groupby(['Patient', 'Sex'])['SmokingStatus'].unique().reset_index()
df_tmp
df_tmp['SmokingStatus'] = df_tmp['SmokingStatus'].str[0]

df_tmp['Sex'] = df_tmp['Sex'].str[0]
df_tmp['SmokingStatus'].value_counts()
df_tmp['Sex'].value_counts()
fig, ax = plt.subplots(1,2, figsize = (20,6), sharex=True)

sns.countplot(x='SmokingStatus',data=df_tmp,ax=ax[0])

sns.countplot(x='SmokingStatus',hue='Sex', data=df_tmp,ax=ax[1])

ax[0].title.set_text('Smoking Status')

ax[1].title.set_text('Smoking Status Vs Sex')

plt.show()
df_test.info()
df_test