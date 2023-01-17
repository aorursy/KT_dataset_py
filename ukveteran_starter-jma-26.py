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
credit = pd.read_csv('../input/creditcard.csv')

credit.head()
plt.matshow(credit.corr())

plt.colorbar()

plt.show()
p = credit.hist(figsize = (20,20))
plt.figure(figsize=(10,7))

sns.scatterplot(x="V1",y='V2',data=credit)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="V1",y='V3',data=credit)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="V2",y='V3',data=credit)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="V1",y='V4',data=credit)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="V2",y='V4',data=credit)

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="V3",y='V4',data=credit)

plt.show()
import matplotlib.pyplot as pl

g = sns.jointplot(x="V1", y="V2", data = credit,kind="kde", color="c")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")
import matplotlib.pyplot as pl

g = sns.jointplot(x="V1", y="V3", data = credit,kind="kde", color="c")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")