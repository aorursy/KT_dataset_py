import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
!mkdir data

!wget -O data/BX-Users.csv 'https://raw.githubusercontent.com/GaoangLiu/figures/master/mlds/data/BX-Users.csv'



users = pd.read_csv('data/BX-Users.csv', sep=';', encoding='latin')

users
def _split(row):

    if ',' in row:

        return [e for e in row.strip().split(',') if e][-1]

    return None



df = users['Location'].apply(_split).dropna()

df.value_counts().nlargest(30)
df.value_counts().nlargest(10).plot()
df.value_counts().nlargest(10).plot(kind='bar')
df.value_counts().nlargest(10).plot(kind='barh').invert_yaxis() 
age = users['Age'].dropna()

age.hist(bins=50)
age = users[users['Age'] >= 120]['Age']

# age.value_counts().sort_index().plot()

age.hist(facecolor='purple', bins=30)