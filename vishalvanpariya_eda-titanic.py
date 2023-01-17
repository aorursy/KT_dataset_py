import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#lets import data from csv file



train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
#Passengerid is a unique identity of passengers so this does not matter to Survival of a Passenger So, let's drop Passengerid feature

train=train.drop(['PassengerId'],1)

train.head()
train['Pclass'].unique()
sns.countplot(x=train['Pclass'])
sns.catplot(x='Pclass',y='Survived',data=train,kind='bar')
#lets see top 5 row data

train['Name'].head()
#let's extract Titles of Passengers

train['Title']=train['Name'].str.extract('([A-Za-z]+)\.',)
train['Title'].unique()
sns.countplot(y=train['Title'])
sns.catplot(y='Title',x='Survived',data=train,kind='bar')
#we do not need Name longer so,Let's drop it

train=train.drop(['Name'],1)

train.head()
sns.countplot(train['Sex'])
sns.catplot(x='Sex',y='Survived',data=train,kind='bar')
sns.catplot(x='Sex',y='Survived',data=train,kind='bar',hue='Pclass')
train['Age_Band']=pd.cut(train['Age'],5)
sns.catplot(y="Age_Band",x='Survived',data=train,kind='bar')
sns.catplot(y="Age_Band",x='Survived',data=train,kind='bar',hue='Pclass')
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train, col='Survived',row='Sex')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train, row='Survived',col='Pclass')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train, row='Survived',col='Embarked')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

grid.add_legend()
train=train.drop(['Age_Band'],1)
train['FamilySize']=train['SibSp']+train['Parch']

train=train.drop(['SibSp','Parch'],1)
sns.catplot(x='FamilySize',y='Survived',data=train,kind='bar')
sns.catplot(x='FamilySize',y='Survived',data=train,kind='bar',hue='Sex')
sns.catplot(x='FamilySize',y='Survived',data=train,kind='bar',hue='Pclass')
train=train.drop(['Ticket'],1)

train.head()
train['Cabin'].isnull().sum()
train['Cabin']=train['Cabin'].fillna('NA')
train.Cabin.unique()
train['Cabin']=train['Cabin'].astype(str).str[0]

train['Cabin'].unique()
sns.catplot(x='Cabin',y='Survived',data=train,kind='bar')
sns.catplot(x='Cabin',y='Survived',data=train,col='Sex',kind='bar')
sns.catplot(x='Cabin',y='Survived',data=train,col='Pclass',kind='bar')
sns.catplot(x='Cabin',y='Survived',data=train,col='Pclass',row='Sex',kind='bar')
print(train['Embarked'].unique())

print(train['Embarked'].isnull().sum())
train['Embarked']=train['Embarked'].fillna(train['Embarked'].value_counts().index[0])

print(train['Embarked'].unique())
sns.catplot(x='Embarked',y='Survived',data=train,kind='bar')
train.head()
corr=train.corr().sort_values(by='Survived',ascending=False).round(2)

plt.subplots(figsize=(8, 6))

sns.heatmap(corr, vmax=.8, square=True);
train.isnull().sum()
print(train.Pclass.unique())

print(train.Sex.unique())
for i in ['male','female']:

    for j in [3,1,2]:

        print(i,j)

        temp_dataset=train[(train['Sex']==i) &  (train['Pclass']==j)]['Age'].dropna()

        print(temp_dataset)

        print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j))

        train.loc[(train.Age.isnull()) & (train.Sex==i) & (train.Pclass==j),'Age']=int(temp_dataset.median())
train.isnull().sum()
train.head()
train=pd.get_dummies(columns=['Pclass','Sex','Cabin','Embarked','Title'],data=train)

train.head()
train['Age_Band']=pd.cut(train['Age'],5)

train['Age_Band'].unique()
train.loc[(train['Age']<=16.136),'Age']=1

train.loc[(train['Age']>16.136) & (train['Age']<=32.102),'Age']=2

train.loc[(train['Age']>32.102) & (train['Age']<=48.068),'Age']=3

train.loc[(train['Age']>48.068) & (train['Age']<=64.034),'Age']=4

train.loc[(train['Age']>64.034) & (train['Age']<=80.),'Age']=5

train['Age'].unique()
train=train.drop(['Age_Band'],1)
train['Fare_Band']=pd.cut(train['Fare'],3)

train['Fare_Band'].unique()
train.loc[(train['Fare']<=170.776),'Fare']=1

train.loc[(train['Fare']>170.776) & (train['Fare']<=314.553),'Fare']=2

train.loc[(train['Fare']>314.553) & (train['Fare']<=513),'Fare']=3

train=train.drop(['Fare_Band'],1)
train.head()
train=pd.get_dummies(columns=['Age','Fare'],data=train)

train.head()