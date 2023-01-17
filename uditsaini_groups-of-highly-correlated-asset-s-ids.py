# Import all the necessary packages 

import kagglegym

import numpy as np

from itertools import chain

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Ridge

import math

import matplotlib.pyplot as plt

%matplotlib inline
# Read the full data set stored as HDF5 file

full_df = pd.read_hdf('../input/train.h5')

mean_values = full_df.median(axis=0)

full_df=full_df.fillna(mean_values)

df=pd.pivot_table(full_df, values='y', index=['timestamp'], columns=['id'], aggfunc=np.sum)

df.to_csv('assets.csv',index=False)

df=pd.read_csv('assets.csv')
cor=df.corr()

cor.loc[:,:] =  np.tril(cor, k=-1)

cor = cor.stack()
ones = cor[cor > 0.5].reset_index().loc[:,['level_0','level_1']]

ones = ones.query('level_0 not in level_1')

groups=ones.groupby('level_0').agg(lambda x: set(chain(x.level_0,x.level_1))).values

print('groups of assets which are correlated on y value more then 0.4')

for g,i in zip(groups,range(len(groups))):

    print(i,g)
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin(map(int,list(groups[127][0])))].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin( map(int,list(groups[36][0])))].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin( map(int,list(groups[152][0])))].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin( map(int,list(groups[162][0])))].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin( map(int,list(groups[23][0])))].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()