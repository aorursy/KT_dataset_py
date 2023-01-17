import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
a=pd.read_csv('../input/train.csv')

a.shape
a.head()
a.describe()
a['Age'].fillna(a['Age'].median(), inplace=True)
survived_sex = a[a['Survived']==1]['Sex'].value_counts()

dead_sex = a[a['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])



df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
figure = plt.figure(figsize=(15,8))

plt.hist([a[a['Survived']==1]['Age'], a[a['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 10,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()