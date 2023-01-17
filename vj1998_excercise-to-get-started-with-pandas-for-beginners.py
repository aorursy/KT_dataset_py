import pandas as pd
import numpy as np

data = {'birds': ['Cranes', 'Cranes', 'plovers', 'spoonbills', 'spoonbills', 'Cranes', 'plovers', 'Cranes', 'spoonbills', 'spoonbills'], 'age': [3.5, 4, 1.5, np.nan, 6, 3, 5.5, np.nan, 8, 4], 'visits': [2, 4, 3, 4, 3, 4, 2, 2, 3, 2], 'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data)
df
df.describe()
print(df.head(2))
print(df['birds'])
print(df['age'])
print(df['birds'].iloc[2], df['age'].iloc[2])
print(df['age'].iloc[3], df['age'].iloc[3])
print(df['visits'].iloc[7], df['age'].iloc[7])
df[df['visits'] < 4]
df[df['age'].isnull()]
df[(df['birds'] == 'Cranes') & (df['age'] < 4)]
df[(df['age'] >= 2) & (df['age'] <= 4)]
df[(df['birds'] == 'Cranes') & (df['visits'] > 0)].sum()
df[['age']].mean()
s = df.xs(3)
df.append(s, ignore_index=True)
df.drop([df.index[9]])
df.groupby(df["birds"]).count()
sort_by_age = df.sort_values('age')
print(sort_by_age.head())
print("------------------------------------")
sort_by_vists = df.sort_values('visits',ascending=False)
print(sort_by_vists.head())

df.priority.map(dict(yes=1, no=0))
df.birds.map(lambda x: 'trumpeters' if x=='Cranes' else x)