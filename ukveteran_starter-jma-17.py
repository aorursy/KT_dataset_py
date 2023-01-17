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
auto= pd.read_csv('../input/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv')

auto.head()
p = auto.hist(figsize = (20,20))
plt.matshow(auto.corr())

plt.colorbar()

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="year",y='maxPrice',data=auto)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="year",y='avgPrice',data=auto)

plt.show()