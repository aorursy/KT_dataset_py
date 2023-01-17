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
gp= pd.read_table('../input/2004-2012.tsv')

gp.head()
plt.matshow(gp.corr())

plt.colorbar()

plt.show()
sns.heatmap(gp.corr())
plt.matshow(gp1.corr())

plt.colorbar()

plt.show()
gp1= pd.read_table('../input/2013-2019.tsv')

gp1.head()
sns.heatmap(gp1.corr())
p = gp.hist(figsize = (20,20))
q = gp1.hist(figsize = (20,20))