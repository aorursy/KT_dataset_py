import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



ids = test['PassengerId']
train.hist();
train['Fare'].hist()
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['1', '2', '3', '4', '5', '6', '7', '8']

    categories = pd.cut(df.Age, bins, labels=group_names)

    return df
def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df
def simplify_fare(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 5, 10, 15, 30, 100, 400, 1000)

    group_names = ['1', '2', '3', '4', '5', '6', '7', '8']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    le = preprocessing.LabelEncoder()

    le = le.fit(categories)

    df.Fare = le.transform(categories)

    return df
def generate_names(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df
def drop(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)
def transform(df):

    df =  simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fare(df)

    df = generate_names(df)

    df = drop(df)

    return df



train = transform(train)

test = transform(test)
train.sample(3)
def encode(df):

    features = list(df)

    for f in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df[f])

        df[f] = le.transform(df[f])

    return df

train = encode(train)

test = encode(test)

test.head()
from sklearn.model_selection import train_test_split



x_all = train.drop(['Survived', 'PassengerId'], axis=1)

y_all = train['Survived']



num_test = 0.20

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=42)
x_train.sample(3)
y_train.sample(3)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(x_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions))

print(predictions)
test.sample(3)

predictions = clf.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()