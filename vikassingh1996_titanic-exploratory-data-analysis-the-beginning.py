# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Import Dependencies

%matplotlib inline
# Importing train data

train = pd.read_csv('../input/train.csv')
#let's check head of the dataset

train.head()
#Nuber of rows and columns

train.shape
#Name of the columns

train.columns
train.info()
train.describe()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap = 'flag')
plt.figure(figsize=(8,5))

sns.heatmap(train.corr(), cmap='YlGnBu', annot=True)
# how many passengers survied?

sns.countplot(data=train, x='Survived', palette='Set1')

print(train.Survived.value_counts())
# how many male and female was survived?

sns.countplot(data=train, x='Survived', hue='Sex', palette='Set2')
# Number of passengers was survived from the pclass 

sns.countplot(data=train, x='Survived', hue='Pclass', palette='Set2')
 # Number of siblings/spouses aboard the Titanic

sns.countplot(data=train, x='SibSp', palette='Set3')

print(train.SibSp.value_counts())
#  number of parents/children aboard the Titanic

sns.countplot(data=train, x='Parch', palette='Set3')

print(train.Parch.value_counts())
# Number of passengers abord from C = Cherbourg, Q = Queenstown, S = Southampton

sns.countplot(data=train, x='Embarked', palette='Set3')

print(train.Embarked.value_counts())
sns.pairplot(data=train, hue ='Survived', diag_kind='hist', height= 3, aspect=0.7,palette='bwr')
sns.lmplot(x='Age',y='Survived',data=train,logistic=True, y_jitter=.03, )
plt.figure(figsize=(8,4))

train['Age'].hist(bins=50, color='teal')
plt.figure(figsize=(8,6))

sns.distplot(train['Age'], bins=50, color='teal')
plt.figure(figsize=(8,4))

train['Fare'].hist(bins=50, color='darkred')
plt.figure(figsize=(8,6))

sns.distplot(train['Fare'].dropna(), bins=50, color='darkred')
plt.figure(figsize=(8, 4))

sns.boxplot(x="Pclass", y="Age", data=train,palette='plasma')
plt.figure(figsize=(13, 2))

sns.boxplot(x=train["Fare"],palette='Blues')
plt.figure(figsize=(13, 2))

sns.boxplot(x=train["Age"],palette='Blues')