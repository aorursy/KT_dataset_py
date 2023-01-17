import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
d = pd.read_csv("../input/bus_stops.csv")
d.head()
d1 = d['Transport'].value_counts()
d1
d1 = (d1 / sum(d1))*100
pd.DataFrame(d1)
d2 = d.loc[d['Transport']=='Day bus stop']

d3 = d.loc[d['Transport']=='Night bus stop']

d4 = d.loc[d['Transport']=='Airport bus stop']
d4['District.Name'].value_counts()
d.loc[d['District.Name']=='Eixample'].count()

d5 = d.groupby('District.Name')['Transport'].count()
d5 = (d5/sum(d5))*100
d5
sns.lmplot(x = 'Longitude', y = 'Latitude', data = d, fit_reg = False, hue ='Transport')

plt.show()
sns.lmplot(x = 'Longitude', y = 'Latitude', data = d, fit_reg = False, hue ='District.Name')

plt.show()
sns.jointplot(x = d['Longitude'], y = d['Latitude'], kind = 'scatter')

plt.show()
sns.lmplot(x = 'Longitude', y = 'Latitude', data = d2, fit_reg = False)

sns.lmplot(x = 'Longitude', y = 'Latitude', data = d3, fit_reg = False)

plt.show()