import matplotlib.pyplot as plt # For plot configuration

import numpy as np              # For numerical operations

import pandas as pd             # For database management

import seaborn as sns           # For plotting data easily



import warnings

warnings.filterwarnings("ignore")



sns.set()
dat = pd.read_csv('../input/weather-data-in-india-from-1901-to-2017/Weather Data in India from 1901 to 2017.csv')
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
set1 = ['JAN']

sb.lineplot(data=dat[set1], linewidth=2.5)
set2 = ['FEB']

sb.lineplot(data=dat[set2], linewidth=2.5)
set3 = ['MAR']

sb.lineplot(data=dat[set3], linewidth=2.5)
set4 = ['APR']

sb.lineplot(data=dat[set4], linewidth=2.5)
set5 = ['MAY']

sb.lineplot(data=dat[set5], linewidth=2.5)
set6 = ['JUN']

sb.lineplot(data=dat[set6], linewidth=2.5)
dat_num = dat.select_dtypes(include = ['float64', 'int64'])

dat_num.head()
dat_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)