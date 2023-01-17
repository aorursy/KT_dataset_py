import numpy as np
import pandas as pd
vals1 = np.array([1, None, 3, 4])
vals1
for dtype in ['object', 'int']:
    print("dtype =", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
vals2 = np.array([1, np.nan, 3, 4]) 
vals2.dtype
1 + np.nan
0 *  np.nan
vals2.sum(), vals2.min(), vals2.max()
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
pd.Series([1, np.nan, 2, None])
x = pd.Series(range(2), dtype=int)
x
x[0] = None
x
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]
data.dropna()
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df
df.dropna()
df.dropna(axis='columns')
df[3] = np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='rows', thresh=3)
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
# forward-fill
data.fillna(method='ffill')
# back-fill
data.fillna(method='bfill')
df
df.fillna(method='ffill', axis=1)