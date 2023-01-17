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
outb= pd.read_csv('../input/outbreaks.csv')

outb.head()
p = outb.hist(figsize = (20,20))
plt.matshow(outb.corr())

plt.colorbar()

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="Year",y='Illnesses',data=outb)

plt.show()