#Introduction

import numpy as np

import pandas as pd

#Object Creation

s = pd.Series([1, 3, 5, np.nan, 6, 8])

print(s)



dates = pd.date_range('20130101', periods=6)

print(dates)



df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

print(df)



df2 = pd.DataFrame({'A': 1.,

                    'B': pd.Timestamp('20130102'),

                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),

                    'D': np.array([3] * 4, dtype='int32'),

                    'E': pd.Categorical(["test", "train", "test", "train"]),

                    'F': 'foo'})

print(df2)



print(df2.dtypes)



#Viewing Data

print(df.head())



print(df.index)



print(df.columns)



print(df.describe())



print(df.T)



print(df.sort_index(axis=1, ascending=False))



print(df.sort_values(by='B'))
#Selection



print(df['A'])



print(df[0:3])



print(df['20130102':'20130104'])



print(df.loc[dates[0]])



print(df.loc[:, ['A', 'B']])



print(df.loc['20130102':'20130104', ['A', 'B']])



print(df.loc['20130102', ['A', 'B']])



print(df.loc[dates[0], 'A'])



print(df.at[dates[0], 'A'])
#Selection by position



print(df.iloc[3])



print(df.iloc[3:5, 0:2])



print(df.iloc[[1, 2, 4], [0, 2]])



print(df.iloc[1:3, :])



print(df.iloc[:, 1:3])



print(df.iloc[1, 1])



print(df.iat[1, 1])
#Boolean Indexing



print(df[df.A > 0])



print(df[df > 0])



df2 = df.copy()

df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

print(df2)



print(df2[df2['E'].isin(['two', 'four'])])
#Setting



s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))



print(s1)



df['F'] = s1



df.at[dates[0], 'A'] = 0



df.iat[0, 1] = 0



df.loc[:, 'D'] = np.array([5] * len(df))



print(df)



df2 = df.copy()



df2[df2 > 0] = -df2



print(df2)
#Missing Data



df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

df1.loc[dates[0]:dates[1], 'E'] = 1



print(df1)



print(df1.dropna(how='any'))



print(df1.fillna(value=5))



print(pd.isna(df1))
#Operations



print(df.mean())



print(df.mean(1))



s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

print(s)



print(df.sub(s, axis='index'))



print(df.apply(np.cumsum))



print(df.apply(lambda x: x.max() - x.min()))



s = pd.Series(np.random.randint(0, 7, size=10))

print(s)



print(s.value_counts())



s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

print(s.str.lower())
#Merge



df = pd.DataFrame(np.random.randn(10, 4))



print(df)



pieces = [df[:3], df[3:7], df[7:]]



print(pieces)



print(pd.concat(pieces))



left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})



right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})



print(left)



print(right)



print(pd.merge(left, right, on='key'))



left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})



right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})



print(pd.merge(left, right, on='key'))



df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])



print(df)



s = df.iloc[3]



print(df.append(s, ignore_index=True))
