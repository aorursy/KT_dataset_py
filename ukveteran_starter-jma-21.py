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
wss = pd.read_csv('../input/who_suicide_statistics.csv')

wss.head()
plt.matshow(wss.corr())

plt.colorbar()

plt.show()
p = wss.hist(figsize = (20,20))
wss['year'].value_counts().plot(kind='bar', title='year 	',figsize=(20,8)) 