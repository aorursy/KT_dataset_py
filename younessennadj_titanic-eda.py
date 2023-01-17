# Load in our libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from matplotlib.pyplot import pie, axis, show

plt.style.use('fivethirtyeight')



import plotly.offline as py

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
# Load in the train datasets

train = pd.read_csv('../input/train.csv')



train.head(10)
# Group the population by class and calculate the mean and the count of the survival

train.groupby(by=['Pclass'])['Survived'].agg(['mean','count'])
# Crate 2 subplots using matplotlib

f,ax=plt.subplots(1,2,figsize=(18,8))

# Group sex and survived by sex and count the mean

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

# Set chart 1 title

ax[0].set_title('Survived vs Sex')

# Use Seaborn to plot the count of sex and survived

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
# Crate 2 subplots using matplotlib

f,ax=plt.subplots(1,2,figsize=(18,8))

# Group classes and survived by classes and count the mean

train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])

# Set chart 1 title

ax[0].set_title('Pclass vs Survived')

# Use Seaborn to plot the count of classes and survived

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
# Crate Category plot from seaborn on classes, survival and Sex

sns.catplot(x="Pclass", y="Survived", hue="Sex",

            palette={"male": "g", "female": "r"},

            markers=["^", "o"], linestyles=["-", "--"],

            kind="point", data=train);
# Create Box plot using Seaborn

sns.catplot(x="Sex", y="Age", hue="Survived", kind="box", data=train);
# sum survived grouped by sex

sums = train.Survived.groupby(train.Sex).sum()

axis('equal');

# draw the pie with sums indexes

pie(sums, labels=sums.index);

show()
# Imputation

# Remove all NULLS in the Embarked column

train['Embarked'] = train['Embarked'].fillna('S')



# Remove all NULLS in the Fare column 

train['Fare'] = train['Fare'].fillna(train['Fare'].median())



train.head(5)
# Mapping Embarked One Hot Encoding

train=pd.get_dummies(train, columns=["Embarked"])



train.head(5)
# Mapping Sex to a binnary feature

train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



train.head(5)
# Adding features



# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature that tells the family size of each passenger

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



# Feature that tells if a passenger is alone

train['IsAlone'] = 0

train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

    

train.head(10)
# Feature selection

# Drop some of unuseful coloumns

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked_C','Embarked_Q','Embarked_S']

train = train.drop(drop_elements, axis = 1)

train.head(10)



# Create the heat map of features correlation

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare',u'Has_Cabin',u'FamilySize',u'IsAlone']], hue='Survived',kind='scatter', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])