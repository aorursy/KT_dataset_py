import matplotlib.pyplot as plt # For plot configuration

import numpy as np              # For numerical operations

import pandas as pd             # For database management

import seaborn as sns           # For plotting data easily



import warnings

warnings.filterwarnings("ignore")



sns.set()
dat = pd.read_csv('../input/weight-vs-age-of-chicks-on-different-diets/ChickWeight.csv')

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



cols = ['weight','Time']

sb.pairplot(dat[cols])
set1 = ['weight']

sb.lineplot(data=dat[set1], linewidth=2.5)
set2 = ['Time']

sb.lineplot(data=dat[set2], linewidth=2.5)