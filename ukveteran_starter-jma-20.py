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
mm = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')

mm.head()
plt.matshow(mm.corr())

plt.colorbar()

plt.show()
p = mm.hist(figsize = (20,20))
mm1=mm.drop(mm.columns[[7, 11,15,16,17,18,19]], axis=1) 

mm1.head()
plt.matshow(mm1.corr())

plt.colorbar()

plt.show()
mm1['Region of Incident'].value_counts().plot(kind='bar', title='Region of Incident',figsize=(20,8)) 
mm1['Cause of Death'].value_counts().plot(kind='bar', title='Cause of Death',figsize=(20,8)) 
sns.heatmap(mm1.corr())