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
ld = pd.read_table('../input/XYZCorp_LendingData.txt',parse_dates=['issue_d'],low_memory=False)
ld.head()
plt.matshow(ld.corr())

plt.colorbar()

plt.show()
p = ld.hist(figsize = (20,20))
plt.figure(figsize=(10,7))

sns.scatterplot(x="funded_amnt",y='installment',data=ld)

plt.show()