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
int= pd.read_csv('../input/Integrated.csv')

int.head()
p = int.hist(figsize = (20,20))
plt.matshow(int.corr())

plt.colorbar()

plt.show()
int['20'].value_counts().plot(kind='bar', title='',figsize=(20,8)) 
int['233'].value_counts().plot(kind='bar', title='',figsize=(20,8)) 