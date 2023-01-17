import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm



%matplotlib inline



with pd.HDFStore('../input/train.h5') as train:

    df = train.get('train')
print('Shape : {}'.format(df.shape))
df.head()
len(df.id.unique()) # how many assets (instruments) are we tracking?
len(df.timestamp.unique()) # how many periods?
market_df = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()

market_df.head()
t      = market_df['timestamp']

y_mean = np.array(market_df['y']['mean'])

y_std  = np.array(market_df['y']['std'])

n      = np.array(market_df['y']['len'])



plt.figure()

plt.plot(t, y_mean, '.')

plt.xlabel('timestamp')

plt.ylabel('mean of y')



plt.figure()

plt.plot(t, y_std, '.')

plt.xlabel('timestamp')

plt.ylabel('std of y')



plt.figure()

plt.plot(t, n, '.')

plt.xlabel('timestamp')

plt.ylabel('portfolio size')
simple_ret = y_mean # this is a vector of the mean of asset returns for each timestamp

cum_ret = np.log(1+simple_ret).cumsum()
portfolio_mean = np.mean(cum_ret)

portfolio_std = np.std(cum_ret)

print("portfolio mean periodic return: " + str(portfolio_mean))

print("portfolio std dev of periodic returns: " + str(portfolio_std))
plt.figure()

plt.plot(t, cum_ret)

plt.xlabel('timestamp')

plt.ylabel('portfolio value')
assets_df = df.groupby('id')['y'].agg(['mean','std',len]).reset_index()

assets_df.head()
assets_df = assets_df.sort_values(by='mean')

assets_df.head()
assets_df.tail()
assets_df.describe()
sns.distplot(assets_df['mean'], rug=True, hist=False)
assets_df.corr()
g = sns.PairGrid(assets_df, vars=["mean", "std", "len"])

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)