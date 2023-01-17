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
cb= pd.read_csv('../input/center_backs.csv')

cm = pd.read_csv('../input/center_mid.csv')

jpa = pd.read_csv('../input/just_player_attributes.csv')

ffifa= pd.read_csv('../input/full_fifa18_data.csv')
cb.head()
plt.matshow(cb.corr())

plt.colorbar()

plt.show()
plt.matshow(cm.corr())

plt.colorbar()

plt.show()
plt.matshow(jpa.corr())

plt.colorbar()

plt.show()
plt.matshow(ffifa.corr())

plt.colorbar()

plt.show()
ffifa.head()
f1=ffifa.drop(ffifa.columns[[4, 6,10]], axis=1) 
f1.head()
f1['nationality'].value_counts().plot(kind='bar', title='Nationality',figsize=(20,8)) 
sns.heatmap(f1.corr())