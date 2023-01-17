# Import all the necessary packages 

import kagglegym

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Ridge

import math

import matplotlib.pyplot as plt



%matplotlib inline

# Read the full data set stored as HDF5 file

full_df = pd.read_hdf('../input/train.h5')
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

rows = np.random.choice(full_df.id, 15)

for key, grp in full_df[full_df.id.isin(rows)].groupby(['id']): 

    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))

plt.legend(loc='best')  

plt.title('y distribution')

plt.show()
feature=[col for col in full_df.columns if col not in ['id','timestamp']]
rcParams['figure.figsize'] = 8, 5

for col in feature:

    rows = np.random.choice(full_df.id, 15)

    for key, grp in full_df[full_df.id.isin(rows)].groupby(['id']): 

        plt.plot(grp['timestamp'], grp[col], label = "id {0:02d}".format(key))

        plt.legend(loc='best')    

        plt.title('features-'+col)

    plt.show()