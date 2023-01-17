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
candy = pd.read_csv('../input/candy.csv')

candy.head()
plt.matshow(candy.corr())

plt.colorbar()

plt.show()
p = candy.hist(figsize = (20,20))
ign=pd.read_csv('../input/ign_scores.csv')

ign.head()
plt.matshow(ign.corr())

plt.colorbar()

plt.show()
p = ign.hist(figsize = (20,20))
mv=pd.read_csv('../input/museum_visitors.csv')

mv.head()
plt.matshow(mv.corr())

plt.colorbar()

plt.show()
p = mv.hist(figsize = (20,20))