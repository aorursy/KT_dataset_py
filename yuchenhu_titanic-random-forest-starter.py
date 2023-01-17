#Basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import string
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
#Make a copy, leave the target column (Survived) aside, and drop the useless features (PassengerId

#and Ticket do not provide any predictive power, otherwise it's counterintuitive)

train_c = train.copy()

test_c = test.copy()

label = train_c.Survived

train_c.drop(['PassengerId', 'Ticket', 'Survived'], inplace=True, axis=1)

test_c.drop(['PassengerId', 'Ticket'], inplace=True, axis=1)

combine = [train_c, test_c]
#Take a look at the summary of the remaining features

train_c.info()
#We need to fill in the null values for both train and test set. Let's deal with Age first. It's

#easy to simply fill the median or mean, but on closer look I bet the passengers' name provide some

#information on Age.

for data in combine:

    data['Title'] = data.Name.apply(lambda x: x.split()[1])

    data.drop('Name', inplace=True, axis=1)
test_c.Title.unique()
#Sort all rare titles into another category, then look at the Age distribution for each title

train_c.Title = train_c.Title.replace(['Planke,', 'Don.', 'Rev.',

       'Billiard,', 'der', 'Walle,', 'Dr.', 'Pelsmaeker,', 'Mulder,', 'y',

       'Steen,', 'Carlo,', 'Mme.', 'Impe,', 'Ms.', 'Major.', 'Gordon,',

       'Messemaeker,', 'Mlle.', 'Col.', 'Capt.', 'Velde,', 'the',

       'Shawah,', 'Jonkheer.', 'Melkebeke,', 'Cruyssen,'], 'Other')

test_c.Title = test_c.Title.replace(['Carlo,', 'Khalil,', 'y', 'Ms.',

       'Palmquist,', 'Col.', 'Planke,', 'Rev.', 'Billiard,',

       'Messemaeker,', 'Dr.', 'Brito,'], 'Other')
#It seems the title does provide some information on age. Let's fill in the Age null values with

#the median of their respective title

for data in combine:

    for title in ['Master.', 'Miss.', 'Mr.', 'Mrs.', 'Other']:

        part = data[data.Title == title]

        median = part['Age'].median()

        data.loc[(data.Title == title) & data['Age'].isnull(), 'Age'] = median
#Now let's deal with Cabin. I bet the first letter provides the most predictive power, so let's

#discard the following numbers

for data in combine:

    data.Cabin = data.Cabin.astype(str).str[0]
#Seems Cabin does have some predictive power on survival rate. The only problem is there are too many

#missing values. The good thing is we haven't lost any information by converting missing values to 

#another category(n)

label.groupby(train_c.Cabin).mean()
#For the two missing values in Embarked, simply fill in the mode

train_c.Embarked.value_counts()

train_c.Embarked = train_c.Embarked.fillna('S')
#There is only one missing Fare value in test set. Simply fill in the median

median = test_c.Fare.median()

test_c.Fare = test_c.Fare.fillna(median)
#Now everything is in place. Next we need to convert all non-numeric values into numeric

train_c.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in ['Sex', 'Cabin', 'Embarked', 'Title']:

    le.fit(list(train_c[column].values)+list(test_c[column].values))

    train_c[column] = le.transform(train_c[column])

    test_c[column] = le.transform(test_c[column])
#Now everything is numeric. The good thing about Random Forest is that we don't need to make strong

#assumptions like the values have equal intervals

train_c
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier()

cross_val_score(rf, train_c, label, cv=3)
#To select the best parameters, we need to grid search

from sklearn.model_selection import GridSearchCV

parameters = {'max_features': ['auto', 'log2', None], 'max_depth': np.arange(5, 13),

              'min_samples_leaf': np.arange(2, 10)}

clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)

clf.fit(train_c, label)
#We will use the grid-searched parameters, but grow more trees, i.e. take a larger n_estimator

clf.best_params_
rf_tuned = RandomForestClassifier(n_estimators=1000, max_depth=10, 

                                  max_features=None, min_samples_leaf=3)

cross_val_score(rf_tuned, train_c, label, cv=3)
rf_tuned.fit(train_c, label)

prediction = rf_tuned.predict(test_c)

submission = pd.DataFrame()

submission['PassengerId'] = test.PassengerId

submission['Survived'] = prediction

submission.to_csv('rf.csv', index=False)