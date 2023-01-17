# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
t=pd.read_csv('/kaggle/input/titanicdataset/Titanic1.csv')
t.head()
t=t.drop(columns=['Cabin'])
t.info()
t.shape
t['SibSp'].value_counts()
t['Parch'].value_counts()
t['Survived'].mean()
t.groupby('Pclass')['Survived'].value_counts()
t.groupby('Pclass')['Survived'].mean()
t.groupby('Sex')['Survived'].mean()
t.groupby(['Pclass','Sex'])['Survived'].mean()
t.groupby('Parch')['Survived'].mean()
t['Age'].describe()
t[t.Fare==0].describe()
t['relatives']=t['Parch']+t['SibSp']
t['relatives'].min()
t['relatives'].max()
t[t.relatives==10]
t['family']=[1 if x>0 else 0for x in t['relatives']]
t['family']
t.isnull().sum()
t['age_group']=['<13' if x <=13 else '13-18' if x <18 else '18-40' if x  <40 else '40-60' if x  <60 else '60-80' if x  <=80 else '80+'if x >80 else "unknown" for x in t['Age']]
t['age_group'].value_counts()
pd.crosstab(t.Pclass,t.age_group)
f=t[['Pclass','age_group']][t.Survived==0]
pd.crosstab(f.Pclass,f.age_group)
e=t[['Pclass','age_group']][t.Survived==1]
import matplotlib.pyplot as plt
plt.xlabel('Age')

plt.hist(t['Age'])

plt.show()
plt.xlabel('Fare')

plt.hist(t['Fare'])

plt.show()
plt.xlabel('Pclass')

plt.hist(t['Pclass'],color='pink')

plt.show()
fig =plt.figure(figsize=(9,9))

ax1 =fig.add_subplot(221)

plt.xlabel('Age')

plt.hist(t['Age'])

ax2=fig.add_subplot(222)

plt.hist(t['Fare'],color='r')

plt.suptitle('two subplots',size=20)

plt.tight_layout(pad=4)

plt.show()
t.boxplot(column=['Age'],grid=False)
sns.countplot(x='Survived',data=t)
sns.catplot(x='Pclass',data=t,hue='Sex',kind='count')
sns.catplot(x='Pclass',data=t,hue='Sex',col='Survived',kind='count')
sns.catplot(x='Pclass',data=t,hue='age_group',kind='count')
sns.catplot(x='Pclass',data=t,y='Age',kind='box')
sns.catplot(x='Pclass',data=t,y='Fare',kind='box')

sns.catplot(x='Pclass',data=t,y='Fare',hue='Sex',kind='box')
sns.catplot(x='Pclass',data=t,y='Fare',hue='Sex',kind='box')