import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (12, 8)
df_train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col=['date'])
df_train.shape, df_test.shape
df_train.head()
num_stores = len(df_train['store'].unique())
fig, axes = plt.subplots(num_stores, figsize=(8, 16))

for s in df_train['store'].unique():
    t = df_train.loc[df_train['store'] == s, 'sales'].resample('W').sum()
    ax = t.plot(ax=axes[s-1])
    ax.grid()
    ax.set_xlabel('')
    ax.set_ylabel('sales')
fig.tight_layout();
months = df_train.index.month
dummy_months = pd.get_dummies(months)
dummy_months.columns = ['month-%s' % m for m in dummy_months.columns]
dummy_months.index = df_train.index

df_train['prev_quarter'] = df_train['sales'].shift(91)
df_train['prev_year'] = df_train['sales'].shift(364)
df_train = pd.concat([df_train, dummy_months], axis=1)
df_train.dropna(inplace=True)
df_train.head()
import datetime

df_test_ex = df_test.copy()
months = df_test_ex.index.month
dummy_months = pd.get_dummies(months)
dummy_months.columns = ['month-%s' % m for m in dummy_months.columns]
dummy_months.index = df_test_ex.index

prev_quarter_dates = df_test_ex.index - datetime.timedelta(days=91)
prev_year_dates = df_test_ex.index - datetime.timedelta(days=364)
df_test_ex['prev_quarter'] = df_train.loc[df_train.index.isin(prev_quarter_dates), 'sales'].values
df_test_ex['prev_year'] = df_train.loc[df_train.index.isin(prev_year_dates), 'sales'].values
df_test_ex = pd.concat([df_test_ex, dummy_months], axis=1)
for m in range(4, 13):
    df_test_ex['month-%s' % m] = 0
df_test_ex.head()
%%time
s1i1 = df_train.loc[(df_train['store'] == 1) & (df_train['item'] == 1), 'sales']
exog_s1i1 = df_train.loc[(df_train['store'] == 1) & (df_train['item'] == 1)].drop(['store', 'item', 'sales'], axis=1)


arimax = sm.tsa.statespace.SARIMAX(s1i1.loc['2013-12-31':], exog=exog_s1i1,
                                   enforce_invertibility=False, enforce_stationarity=False, freq='D',
                                   order=(7,0,0)).fit()
print(arimax.summary())
arimax_results = df_test.reset_index()
arimax_results['sales'] = 0
tic = time.time()

for s in arimax_results['store'].unique():
    for i in arimax_results['item'].unique():
        endog = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i), 'sales']
        exog = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i)].drop(['store', 'item', 'sales'], axis=1)
        arimax = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog,
                                           enforce_invertibility=False, enforce_stationarity=False, freq='D',
                                           order=(7,0,0)).fit()
        exog_test = df_test_ex.loc[(df_test_ex['store'] == s) & (df_test_ex['item'] == i)].drop(['id', 'store', 'item'], axis=1)
        fcst = arimax.predict(start='2017-12-31', end='2018-03-31', dynamic=True, exog=exog_test)
        arimax_results.loc[(arimax_results['store'] == s) & (arimax_results['item'] == i), 'sales'] = fcst.values[1:]
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}m".format(s, i, (toc-tic)/60))
arimax_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
arimax_results.head()
arimax_results.to_csv('arimax_results.csv', index=False)
exog = df_train.loc[(df_train['store'] == 10) & (df_train['item'] == 50)].drop(['store', 'item', 'sales'], axis=1)
forecast = arimax.predict(start='2017-10-01', end='2017-12-31', dynamic=True, exog=exog)
actual = df_train.loc[(df_train['store'] == 10) & (df_train['item'] == 50), 'sales']

forecast.plot()
actual.loc['2017-10-01':].plot()
plt.legend(['ARIMAX', 'Actual'])
plt.ylabel('Sales');
