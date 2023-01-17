import pandas as pd
from pandas import Series, DataFrame
titanic_df = pd.read_csv(r'C:\Users\charu\Desktop\train.csv')
titanic_df.head()
titanic_df.info()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.factorplot('Sex',data=titanic_df, kind='count')
sns.factorplot('Sex',data=titanic_df, hue='Pclass', kind='count')
sns.factorplot('Pclass',data=titanic_df, hue='Sex', kind='count')
def male_female_child(passenger):
    age,sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis=1)
titanic_df[0:10]
sns.factorplot('Pclass',data=titanic_df, hue='person',kind='count')
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['person'].value_counts()
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim= (0, oldest))

fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim= (0, oldest))

fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim= (0, oldest))

fig.add_legend()
titanic_df.head()
deck = titanic_df['Cabin'].dropna()
deck.head()
levels = []

for level in deck:
    
    levels.append(level[0])
    
cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.factorplot('Cabin', data=cabin_df,palette='winter_d', kind='count')
cabin_df = cabin_df[cabin_df.Cabin != 'T' ]
sns.factorplot('Cabin', data=cabin_df,palette='summer', kind='count')
titanic_df.head()
sns.factorplot('Embarked', data = titanic_df, hue='Pclass', order = ['C','Q','S'], kind='count')


#Who was alone and who was with family
titanic_df.head()
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone']
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

titanic_df.head()
sns.factorplot('Alone', data=titanic_df, palette='Blues', kind='count')
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})
sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')
sns.factorplot('Pclass', 'Survived',hue='person', data=titanic_df)
sns.lmplot('Age','Survived',data=titanic_df)
sns.lmplot('Age','Survived', hue='Pclass', palette='winter', data=titanic_df)
generations = [10,20,40,60,80]
sns.lmplot('Age','Survived', hue='Pclass', palette='winter', data=titanic_df, x_bins=generations)
sns.lmplot('Age','Survived', hue='Sex', palette='winter', data=titanic_df, x_bins=generations)
