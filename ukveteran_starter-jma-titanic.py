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
train = pd.read_csv("../input/train.csv")

validation = pd.read_csv("../input/test.csv")

train.head()
plt.matshow(train.corr())

plt.colorbar()

plt.show()
p = train.hist(figsize = (20,20))
import matplotlib.pyplot as pl

g = sns.jointplot(x="V1", y="V2", data = credit,kind="kde", color="c")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")
sns.countplot(x=train['Sex'])

fig=plt.gcf()

fig.set_size_inches(6,4)
sns.countplot(x=train['Survived'])

fig=plt.gcf()

fig.set_size_inches(6,4)
plt.matshow(validation.corr())

plt.colorbar()

plt.show()
validation.head()
sns.countplot(x=validation['Sex'])

fig=plt.gcf()

fig.set_size_inches(6,4)
sns.countplot(x=validation['Pclass'])

fig=plt.gcf()

fig.set_size_inches(6,4)