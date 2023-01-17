import matplotlib.pyplot as plt # For plot configuration

import numpy as np              # For numerical operations

import pandas as pd             # For database management

import seaborn as sns           # For plotting data easily



import warnings

warnings.filterwarnings("ignore")



sns.set()
dat = pd.read_csv('../input/air-quality-data/airmay.csv')

n_rows, n_cols = dat.shape

print('The dataset has {} rows and {} columns.'.format(n_rows, n_cols))

dat.head()
columns = dat.columns

print('Columns names: {}.'.format(columns.tolist()))
sns.pairplot(data=dat, 

             kind='reg')
corr = dat.corr()

corr.style.background_gradient()

corr.style.background_gradient().set_precision(2)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sb

sb.set(style="darkgrid")



cols = ['X1', 'X2', 'X3', 'Y']

sb.pairplot(dat[cols])
set1 = ['X1']

sb.lineplot(data=dat[set1], linewidth=2.5)
set2 = ['X2']

sb.lineplot(data=dat[set2], linewidth=2.5)
set3 = ['X3']

sb.lineplot(data=dat[set3], linewidth=2.5)
set4 = ['Y']

sb.lineplot(data=dat[set4], linewidth=2.5)