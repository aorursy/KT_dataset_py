import pandas as pd



stocks = pd.read_csv('../input/stocks.csv', header='infer' ) 

stocks.index = stocks['Date']

stocks = stocks.drop(['Date'],axis=1)

stocks.head()
import numpy as np



N,d = stocks.shape

delta = pd.DataFrame(100*np.divide(stocks.iloc[1:,:].values-stocks.iloc[:N-1,:].values, stocks.iloc[:N-1,:].values),

                    columns=stocks.columns, index=stocks.iloc[1:].index)

delta.head()
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

%matplotlib inline



fig = plt.figure(figsize=(8,5)).gca(projection='3d')

fig.scatter(delta.MSFT,delta.F,delta.BAC)

fig.set_xlabel('Microsoft')

fig.set_ylabel('Ford')

fig.set_zlabel('Bank of America')

plt.show()
meanValue = delta.mean()

covValue = delta.cov()

print(meanValue)

print(covValue)
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6))



ts = delta[440:447]

ts.plot.line(ax=ax1)

ax1.set_xticks(range(7))

ax1.set_xticklabels(ts.index)

ax1.set_ylabel('Percent Change')



ts = delta[568:575]

ts.plot.line(ax=ax2)

ax2.set_xticks(range(7))

ax2.set_xticklabels(ts.index)

ax2.set_ylabel('Percent Change')
fig = plt.figure(figsize=(10,4))



ax = fig.add_subplot(111)

ts = delta[445:452]

ts.plot.line(ax=ax)

ax.set_xticks(range(7))

ax.set_xticklabels(ts.index)

ax.set_ylabel('Percent Change')