import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (16, 9)
df_train = pd.read_csv('../input/train.csv', parse_dates=[0])
df_test = pd.read_csv('../input/test.csv', parse_dates=[0])
df_train.shape, df_test.shape
df_train.head()
for s in df_train['store'].unique():
    for i in df_train['item'].unique():
        t = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i)]
        plt.plot_date(t['date'], t['sales'], alpha=0.1)
plt.grid()
plt.xlabel('date')
plt.ylabel('sales');
am_results = df_test.copy()
am_results['sales'] = 0

for s in am_results['store'].unique():
    for i in am_results['item'].unique():
        historical_average = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i), 'sales'].mean()
        am_results.loc[(am_results['store'] == s) & (am_results['item'] == i), 'sales'] = historical_average
am_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
am_results.head()
am_results.to_csv('am_results.csv', index=False)
snm_results = df_test.copy()
snm_results['date'] = snm_results['date'].astype(np.datetime64)
snm_results['sales'] = 0
import datetime

prev_dates = snm_results.loc[(snm_results['store'] == 1) & (snm_results['item'] == 1), 'date'] - datetime.timedelta(days=365)
for s in snm_results['store'].unique():
    for i in snm_results['item'].unique():
        snm_results.loc[(snm_results['store'] == s) & (snm_results['item'] == i), 'sales'] = \
            df_train.loc[((df_train['store'] == s) & (df_train['item'] == i)) & (df_train['date'].isin(prev_dates)), 'sales'].values
snm_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
snm_results.head()
snm_results.to_csv('snm_results.csv', index=False)
