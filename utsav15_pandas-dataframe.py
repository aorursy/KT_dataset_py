import pandas as pd
import numpy as np
from numpy.random import randn
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df
df['W']
# Pass a list of column names
df[['W','Z']]
type(df['W'])
df['new'] = df['W'] + df['Y']
df
df.drop('new',axis=1)
# Not inplace unless specified!
df
df.drop('new',axis=1,inplace=True)
df
df.drop('E',axis=0)
df.loc['A']
df.iloc[2]
df.loc['B','Y']
df.loc[['A','B'],['W','Y']]
df
df>0
df[df>0]
df.loc[df['W']>0]
df.loc[df['W']>0]['Y']
df.loc[df['W']>0][['Y','X']]
df
# Reset to default 0,1...n index
df.reset_index()
newind = 'CA NY WY OR CO'.split()
df['States'] = newind
df
df.set_index('States')
df
df.set_index('States',inplace=True)
df
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
hier_index
df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df
df.loc['G1']
df.index.names = ['Group','Num']
df
df.loc['G1']