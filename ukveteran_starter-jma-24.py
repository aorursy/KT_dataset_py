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
forest = pd.read_csv('../input/covtype.csv')

forest.head()
plt.matshow(forest.corr())

plt.colorbar()

plt.show()
p = forest.hist(figsize = (20,20))