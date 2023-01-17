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
aas = pd.read_csv('../input/age_specific_fertility_rates.csv')

bd = pd.read_csv('../input/birth_death_growth_rates.csv')

me = pd.read_csv('../input/mortality_life_expectancy.csv')

myp= pd.read_csv('../input/midyear_population.csv')
plt.matshow(aas.corr())

plt.colorbar()

plt.show()
plt.matshow(bd.corr())

plt.colorbar()

plt.show()
plt.matshow(me.corr())

plt.colorbar()

plt.show()
plt.matshow(myp.corr())

plt.colorbar()

plt.show()
p = aas.hist(figsize = (20,20))
p =bd.hist(figsize = (20,20))
p = me.hist(figsize = (20,20))
p = myp.hist(figsize = (20,20))