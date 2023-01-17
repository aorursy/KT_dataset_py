# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df  = pd.read_csv("/kaggle/input/titanic/train.csv")
print ("*"*10, "Dataset information", "*"*10)
print (train_df.info())
print ("*"*10, "First 5 test rows", "*"*10)
print (train_df.head(5))
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
print ("*"*10, "Dataset information", "*"*10)
print (test_df.info())
print ("*"*10, "Last 5 test rows", "*"*10)
print (test_df.tail(5))
train_df["Age"].isnull().sum()
#Age null fix
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    
    
    dataset['Age'] = dataset['Age'].fillna(mean)
    dataset["Age"] = train_df["Age"].astype(float)
train_df["Age"].isnull().sum()


#Data processing
#1. In the cabin variable, create new column and add there only first letters of the column
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Deck'] = dataset['Cabin'].fillna("U")
    dataset['Deck'] = dataset['Cabin'].astype(str).str[0] 
    dataset['Deck'] = dataset['Deck'].str.capitalize()
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int) 

train_df['Deck'].value_counts()

#Dropping Cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df.drop(['Cabin'], axis=1, inplace = True)
print (train_df['Deck'].value_counts())
train_df.info()
train_df.Sex.value_counts()
#print (train_df['Embarked'].value_counts())

common_value = 'S'
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

train_df.info()
train_df.head(10)
data = [train_df, test_df] 
embarkedMap = {"S": 0, "C": 1, "Q": 2}

genderMap = {"male": 0, "female": 1}

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)
    dataset['Sex'] = dataset['Sex'].map(genderMap)
    
train_df['Sex'].describe()
train_df['Embarked'].describe()
train_df.info()
train_df.head(10)

#Title extraction
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
    
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme'], 'Rare')

    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    
    
print (train_df['Title'].value_counts())


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df['Title'].value_counts()


train_df.info()
train_df['Ticket'].value_counts()
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1) 

data = [train_df, test_df]
for dataset in data:
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
# let's see how it's distributed 
train_df['Age'].value_counts()

data = [train_df, test_df]



for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(float)

train_df['Fare'].value_counts()

train_df.info()
'''
train_df  = train_df.drop("Ticket", axis=1)
test_df  = test_df.drop("Ticket", axis=1)
'''
train_df = train_df.drop(['PassengerId'], axis=1)
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

test_df.head(10)
X_test = test_df.drop("PassengerId", axis=1).copy()
X_test.head(10)
X_train.info()
#  Decision tree

from sklearn import tree
clf_dt = tree.DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=25,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
clf_dt.fit(X_train, Y_train)

Y_pred_svm  = clf_dt.predict(X_test)

scores_dt = cross_val_score(clf_dt, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_dt.mean())