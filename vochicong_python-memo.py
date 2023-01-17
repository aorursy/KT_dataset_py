import pandas as pd
s = pd.Categorical(['b','a','b','c',None,'b','c'], ordered=True, categories=['a', 'b', 'c'])
print("Series {0}, codes {1}".format(s, s.codes))
s = pd.Categorical(['b','a','b','c',None,'b','c'], ordered=True, categories=['b', 'c'])
print("Series {0}, codes {1}".format(s, s.codes))
s = pd.Categorical(['b','a','b','c',None,'b','c'], ordered=True, categories=['b', 'c', 'd'])
print("Series {0}, codes {1}".format(s, s.codes))
pd.Series(s).cat.codes
df = pd.DataFrame([['green', 'M', 10.1],
                   [None, 'XL', 14.3],
                   ['red', 'L', 13.5],
                   ['blue', 'XL', 15.3]])
df.columns = ['color', 'size', 'price']
df
size = pd.Categorical(df['size'], ordered=True, categories=['M', 'L', 'XL'] )
df['size_cat'] = pd.Series(size.codes)
df
dummies = pd.get_dummies(df['color'], dummy_na=True, drop_first=True)
dummies
df = pd.concat([df, dummies], axis=1, sort=False)
df.drop(['color', 'size'], axis=1)
df_log = pd.DataFrame(columns=['option1', 'option2'])
df_log
df_log = df_log.append(pd.DataFrame({'option1': [3], 'option2': [7]}))
df_log
df_log = df_log.append(pd.DataFrame([[4,8]], columns=['option1', 'option2']),
                       ignore_index=True)
df_log
dict = {'a': 1, 'b': 2, 'c': 3}
{k: dict.get(k) for k in ['b', 'c', 'd']}
dict['d'] = 4
{k: dict.get(k) for k in ['b', 'c', 'd']}
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                  'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                  'baz': [1, 2, 3, 4, 5, 6],
                  'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
df
df.pivot(index='foo', columns='bar', values='baz')
df.pivot(index='foo', columns='bar')['baz']
df.pivot(index='foo', columns='bar')
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                  'bar': ['A', 'A', 'C', 'A', 'B', 'C'],
                  'baz': [1, 2, 3, 4, 5, 6],
                  'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
df.pivot(index='foo', columns='bar', values='baz') # raise exception about duplicate index in 'foo' and 'bar'
