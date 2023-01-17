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
chen= pd.read_csv('../input/chennai_reservoir_levels.csv')

chen.head()
p = chen.hist(figsize = (20,20))
plt.matshow(chen.corr())

plt.colorbar()

plt.show()
chen1= pd.read_csv('../input/chennai_reservoir_rainfall.csv')

chen1.head()
q = chen1.hist(figsize = (20,20))
plt.matshow(chen1.corr())

plt.colorbar()

plt.show()