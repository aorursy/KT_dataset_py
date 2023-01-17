import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score

%matplotlib inline



#import warnings

#warnings.filterwarnings("ignore", category=FutureWarning)



import os

print(os.listdir("../input"))
titanic = pd.read_csv("../input/train.csv")

titanic.head()
titanic.describe()
titanic.groupby('Survived').mean()
titanic.groupby(titanic['Age'].isnull()).mean()
for i in ['Age', 'Fare']:

    died = list(titanic[titanic['Survived'] == 0][i].dropna())

    survived = list(titanic[titanic['Survived'] == 1][i].dropna())

    xmin = min(min(died), min(survived))

    xmax = max(max(died), max(survived))

    width = (xmax - xmin) / 40

    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))

    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))    

    plt.legend(['Did not survived', 'Survived'])

    plt.title('Overlaid histogram for {}'.format(i))

    plt.show()
for i, col in enumerate(['Pclass', 'SibSp', 'Parch']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2)
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']

titanic.drop(['SibSp', 'Parch'], axis=1, inplace=True)

sns.catplot(x='Family_cnt', y='Survived', data=titanic, kind='point', aspect=2)

titanic.head()
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

titanic.isnull().sum()
titanic.head(10)
titanic.info()
titanic.groupby(titanic['Cabin'].isnull()).mean()
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)

titanic.drop(['Cabin'], axis=1, inplace=True)

titanic.head(10)
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2)
titanic.pivot_table('Survived', index='Sex', columns='Embarked', aggfunc='count')
titanic.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count')
titanic.head()
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
titanic.head()
gender_num = {'male': 0, 'female': 1}

titanic['Sex'] = titanic['Sex'].map(gender_num)

titanic.head()
features = titanic.drop('Survived', axis=1)

labels = titanic['Survived']



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
for dataset in [y_train, y_val, y_test]:

    print(round(len(dataset) / len(labels), 2))
rf = RandomForestClassifier()



scores = cross_val_score(rf, X_train, y_train.values.ravel(), cv=5)

scores
def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))

    

    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
parameters = {

    'n_estimators': [5, 50, 100],

    'max_depth': [2, 10, 20, None]

}



cv = GridSearchCV(rf, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)
rf1 = RandomForestClassifier(n_estimators=5, max_depth=10)

rf1.fit(X_train, y_train.values.ravel())



rf2 = RandomForestClassifier(n_estimators=50, max_depth=10)

rf2.fit(X_train, y_train.values.ravel())



rf3 = RandomForestClassifier(n_estimators=100, max_depth=None)

rf3.fit(X_train, y_train.values.ravel())
for mdl in [rf1, rf2, rf3]:

    y_pred = mdl.predict(X_val)

    accuracy = round(accuracy_score(y_val, y_pred), 3)

    precision = round(precision_score(y_val, y_pred), 3)

    recall = round(recall_score(y_val, y_pred), 3)

    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,

                                                                        mdl.n_estimators,

                                                                        accuracy,

                                                                        precision,

                                                                        recall))
y_pred = rf2.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)

precision = round(precision_score(y_test, y_pred), 3)

recall = round(recall_score(y_test, y_pred), 3)

print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rf2.max_depth,

                                                                    rf2.n_estimators,

                                                                    accuracy,

                                                                    precision,

                                                                    recall))
test = pd.read_csv("../input/test.csv")

test.head()
gender_num = {'male': 0, 'female': 1}

test['Sex'] = test['Sex'].map(gender_num)

test['Age'].fillna(test['Age'].mean(), inplace=True)

test['Family_cnt'] = test['SibSp'] + test['Parch']

test['Cabin_ind'] = np.where(test['Cabin'].isnull(), 0, 1)



passengerId = test['PassengerId']

test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

test.head()

test.info()
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
survived = rf2.predict(test)
data = {'PassengerId': passengerId, 'Survived': survived}

final = pd.DataFrame(data)
final.head()
final.to_csv('submission.csv', index=False)