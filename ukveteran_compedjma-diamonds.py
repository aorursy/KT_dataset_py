import matplotlib.pyplot as plt # For plot configuration

import numpy as np              # For numerical operations

import pandas as pd             # For database management

import seaborn as sns           # For plotting data easily



import warnings

warnings.filterwarnings("ignore")



sns.set()
dat = pd.read_csv('../input/diamonds/diamonds.csv')

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



cols = ['carat','depth', 'table','price', 'x','y', 'z']

sb.pairplot(dat[cols])
set1 = ['carat']

sb.lineplot(data=dat[set1], linewidth=2.5)
set2 = ['depth']

sb.lineplot(data=dat[set2], linewidth=2.5)
set3 = ['table']

sb.lineplot(data=dat[set3], linewidth=2.5)
set4 = ['price']

sb.lineplot(data=dat[set4], linewidth=2.5)
set5 = ['y']

sb.lineplot(data=dat[set5], linewidth=2.5)
set6 = ['z']

sb.lineplot(data=dat[set6], linewidth=2.5)