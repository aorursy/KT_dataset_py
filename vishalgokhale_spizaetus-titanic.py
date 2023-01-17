# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]
print(train_df.columns.names)
print(train_df.columns.values)
train_df.head()
train_df.tail()
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
train_df.describe(percentiles=[0.25,0.61,0.62,0.78])
train_df.describe(percentiles=np.arange(0.1,1.0,0.1))
train_df.describe(include='all', percentiles=np.arange(0.1,1.0,0.1))
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by=['Survived'], ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(['Survived'],ascending=False)
train_df[['Parch','SibSp','Survived']].groupby(['Parch','SibSp'], as_index=False).mean()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist,'Age',bins=20)
grid = sns.FacetGrid(train_df, col='Survived', hue='Survived', row='Pclass', size=3, aspect=1.6)
grid.map(plt.hist,'Age', alpha=0.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df, col='Survived', hue='Survived', row='Sex', size=3, aspect=1.6)
grid.map(plt.hist,'Age', alpha=0.9, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df, col='Survived', hue='Survived', row='SibSp', size=3, aspect=1.6)
grid.map(plt.hist,'Age', alpha=0.8, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df,col='Embarked')
grid.map(sns.pointplot, 'Pclass','Survived','Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df,row='Embarked',size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass','Survived','Sex', palette='Set1')
grid.add_legend()
grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'r', 1: 'b'})
#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6, hue='Survived', palette={0: 'r', 1: 'b'})
grid.map(sns.barplot, 'Sex', 'Age', alpha=0.5689, ci=None)
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6, hue='Survived', palette={0: 'r', 1: 'b'})
grid.map(sns.barplot, 'Sex', 'Age', alpha=0.5, ci=None, order=['female','male'])
grid.add_legend()
someOtherDataFrame = train_df.set_index('Pclass')
someOtherDataFrame.columns.values
someOtherDataFrame.index.all
train_df.columns.values
pClassDf = train_df.set_index('PassengerId')
pClassDf.axes
pClassDf = train_df.set_index('Pclass', drop=False)
pClassDf
pClassDf.loc[3,:]
pClassDf.shape
pClassDf.drop([1,2])
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
str.extract("asdcasd","([A-Za-z]+)\.")
[" asdcad.","asdcad"].extract("([A-Za-z]+)\.", expand=False)
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
