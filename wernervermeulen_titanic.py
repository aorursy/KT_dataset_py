# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading the dataset



train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')     
train.head()
test.head()
train.info()
test.info()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

pd.plotting.register_matplotlib_converters()

# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing Fares of the passengers

sns.lineplot(data=train['Fare'])
# Change the style of the figure to the "dark" theme

sns.set_style("dark")



# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Age vs Fare")



# Line chart showing daily global streams of 'Shape of You'

sns.lineplot(data=train['Age'], label="Age")



# Line chart showing daily global streams of 'Despacito'

sns.lineplot(data=train['Fare'], label="Fare")



# Add label for horizontal axis

#plt.xlabel("Date")
sns.scatterplot(x=train['Age'], y=train['Fare'], hue=train['Sex'])
sns.lmplot(x="Age", y="Fare", hue="Sex", data=train)
sns.swarmplot(x=train['Fare'],

              y=train['Age'])
sns.swarmplot(x=train['Cabin'],

              y=train['Fare'])
# KDE plot 

sns.kdeplot(data=train['Age'], shade=True)
# 2D KDE plot

sns.jointplot(x=train['Age'], y=train['Fare'], kind="kde")
# KDE plots for each species

sns.kdeplot(data=train['Age'], label="Age", shade=True)

sns.kdeplot(data=train['Fare'], label="Fare", shade=True)



# Add title

plt.title("Distribution of Cases, by Age and Fare")
# Histograms for each passenger

sns.distplot(a=train['Age'], label="Age", kde=False)

sns.distplot(a=train['Fare'], label="Fare", kde=False)



# Add title

plt.title("Histogram of Age and Fare")



# Force legend to appear

plt.legend()
# Draw a nested boxplot to show bills by day and time

sns.boxplot(x="Age", y="Fare",

            hue="Sex", palette=["m", "g"],

            data=train)

sns.despine(offset=10, trim=True)
train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)

train['Embarked'].fillna(value='S',inplace=True)



train['family']=train['SibSp']+train['Parch']+1

test['family']=test['SibSp']+train['Parch']+1

train.info()
test.info()
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])

train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])

train.head()

test.info()
train_clean=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])

train_clean.head()

train_clean.info()
test.head(2)
test_clean=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])

test_clean.head()

test_clean.info()
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report

X_train=train.drop(columns=['Survived'])

Y_train=train.Survived

Y_train.head()

num_feat=X_train.select_dtypes(include='number').columns.to_list()

cat_feat=X_train.select_dtypes(include='object').columns.to_list()

num_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('scale', StandardScaler())

])



cat_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('coder', OneHotEncoder(handle_unknown='ignore'))

])



ct=ColumnTransformer(remainder='drop',

    transformers=[

    ('numerical',num_pipe, num_feat),

    ('categorical',cat_pipe, cat_feat)

])



model=Pipeline([

    ('transformer', ct),

    ('predictor', RandomForestClassifier())

])

model.fit(X_train, Y_train);
model.score(X_train, Y_train)
