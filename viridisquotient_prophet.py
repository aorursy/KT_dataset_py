import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from fbprophet import Prophet

pd.options.display.max_columns = 99
plt.rcParams['figure.figsize'] = (16, 9)
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
proph_results = df_test.reset_index()
proph_results['sales'] = 0
tic = time.time()

for s in proph_results['store'].unique():
    for i in proph_results['item'].unique():
        proph_train = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i)].reset_index()
        proph_train.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
        
        m = Prophet()
        m.fit(proph_train[['ds', 'y']])
        future = m.make_future_dataframe(periods=len(df_test.index.unique()), include_history=False)
        fcst = m.predict(future)
        
        proph_results.loc[(proph_results['store'] == s) & (proph_results['item'] == i), 'sales'] = fcst['yhat'].values
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}s".format(s, i, toc-tic))
proph_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
proph_results.head()
proph_results.to_csv('proph_results.csv', index=False)
