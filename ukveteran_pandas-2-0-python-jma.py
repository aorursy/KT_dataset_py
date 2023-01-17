import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

ts = pd.Series(np.random.randn(1000),

                  index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
df = pd.DataFrame(np.random.randn(1000, 4),

                       index=ts.index, columns=list('ABCD'))
df = df.cumsum()
plt.figure()
df.plot()
df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df2.plot.bar()

df2.plot.bar(stacked=True)

df2.plot.barh(stacked=True)
df3 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df3.plot.area()
df4 = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

df4['b'] = df4['b'] + np.arange(1000)

df4.plot.hexbin(x='a', y='b', gridsize=25)
df5 = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

df5['b'] = df5['b'] = df5['b'] + np.arange(1000)

df5['z'] = np.random.uniform(0, 3, 1000)

df5.plot.hexbin(x='a', y='b', C='z', reduce_C_function=np.max, gridsize=25)