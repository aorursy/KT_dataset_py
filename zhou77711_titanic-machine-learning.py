import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import os

from pathlib import Path

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import (GradientBoostingClassifier,

                              RandomForestClassifier,

                              AdaBoostClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier
path = Path('../input')

train_df = pd.read_csv(path/'train.csv')

print(len(train_df))

print(train_df.head())
test_df = pd.read_csv(path/'test.csv')

print(len(test_df))

print(test_df.head())
train_df.Embarked = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

train_df.Embarked = train_df.Embarked.map({'S':0, 'C':1, 'Q':2}).astype(int)
test_df.Embarked = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

test_df.Embarked = test_df.Embarked.map({'S':0, 'C':1, 'Q':2}).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_pid = test_df.PassengerId
print(train_df.Pclass.min(), train_df.Pclass.max())
df = pd.concat([train_df, test_df], ignore_index=True)

print(len(df))

print(df.head())
guess_ages = np.zeros((2,3)) #Store the guess for each  Pclass and Sex

for dataset in [train_df, test_df]:

    for i in range(0, 2): # Sex has two values: 0 and 1

        for j in range(0, 3): # Pclass range from 1 to 3

            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()

    #         print(i,j,guess_df)

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)
print(df.Age.max(), df.Age.min())
for dataset in [train_df, test_df]:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
print(df.Fare.max(), df.Fare.min())
for dataset in [train_df, test_df]:

    dataset.loc[ dataset['Fare'] <= 128, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 128) & (dataset['Fare'] <= 256), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 256) & (dataset['Fare'] <= 384), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 384, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df.head()
train_df = train_df.drop(['Name', 'Ticket', 'Cabin','PassengerId'], axis=1)

train_df.head()
test_passengerID = test_df.PassengerId

test_data = test_df.drop(['Name', 'Ticket', 'Cabin','PassengerId'], axis = 1)

test_data.head()
def split_by_rand_pct(idx_list:list, valid_pct:float=0.2, seed:int=None)->'List':

    "Split the items randomly by putting `valid_pct` in the validation set, optional `seed` can be passed."

    if valid_pct==0.: return

    if seed is not None: np.random.seed(seed)

    rand_idx = np.random.permutation(idx_list)

    cut = int(valid_pct * len(idx_list))

    return (rand_idx[cut:].tolist(), rand_idx[:cut].tolist())
train_data_idx, valid_data_idx = split_by_rand_pct(list(train_df.index),seed=2019)
train_data = train_df.loc[train_data_idx,:]

train_label = list(train_data.Survived)

train_data = train_data.drop(['Survived'],axis=1)

print(train_data.head())

print(len(train_data))
valid_data = train_df.loc[valid_data_idx,:]

valid_label = list(valid_data.Survived)

valid_data = valid_data.drop(['Survived'],axis=1)

valid_data.head()
class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        if seed: param['random_state'] = seed

        self.clf = clf(**params)

    

    def fit(self, data, label):

        return self.clf.fit(data, label)

    

    def predict(self, data):

        return self.clf.predict(data)

    

    def score(self, data, label):

        return self.clf.score(data, label)
# Random Forest parameters

rf_params_1 = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}

rf_params_2 = {

    'max_depth': 5, 

    'n_estimators': 10, 

    "max_features": 1

}

# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params_1 = {

    'kernel' : 'linear',

    'C' : 0.025

    }

svc_params_2 = {

    'gamma' : 2,

    'C' : 1

    }



# KNN parameters

knn_params_1 = {

    'n_neighbors': 2

}

knn_params_2 = {

    'n_neighbors': 3

}

# DecisionTreeClassifier parameters

dt_params = {

    'max_depth': 5

}



# LinearSVC

lsvc_params = {}



# LogisticRegression

lr_params = {}



# GaussianNB

nb_params = {}



# SGDClassifier

sgd_params = {}
# Create 5 objects that represent our 4 models

SEED = 0

rf_1 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_1)

rf_2 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_2)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc_1 = SklearnHelper(clf=SVC, seed=SEED, params=svc_params_1)

svc_2 = SklearnHelper(clf=SVC, seed=SEED, params=svc_params_2)

knn_1 = SklearnHelper(clf=KNeighborsClassifier, seed=SEED, params=knn_params_1)

knn_2 = SklearnHelper(clf=KNeighborsClassifier, seed=SEED, params=knn_params_2)

dt =  SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params)

lsvc =  SklearnHelper(clf=LinearSVC, seed=SEED, params=lsvc_params)

lr =  SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)

nb =  SklearnHelper(clf=GaussianNB, seed=SEED, params=nb_params)

sgd =  SklearnHelper(clf=SGDClassifier, seed=SEED, params=sgd_params)
from sklearn.metrics import roc_auc_score

from sklearn import metrics

curve_dict = {}

algos = [rf_1,rf_2, ada, gb, svc_1, svc_2, knn_1, knn_2, dt, lsvc, lr, nb, sgd]

for model in algos:

    model.fit(train_data, train_label)

    valid_pred = model.predict(valid_data)

    acc = round(model.score(valid_data, valid_label) * 100, 2)

    auc = roc_auc_score(valid_label, valid_pred)

    fpr, tpr, thresholds = metrics.roc_curve(valid_label, valid_pred, pos_label=1)

    curve_dict[algos.index(model)] = (fpr, tpr, thresholds,auc,acc)
curve_dict
from matplotlib import pyplot

pyplot.plot([0, 1], [0, 1], linestyle='--')

for i in curve_dict.keys():

    pyplot.plot(curve_dict[i][0], curve_dict[i][1], marker='.')

pyplot.show()
import math

dist = 30

ind = []

for i in curve_dict.values():

    tmp = math.sqrt(i[0][1]**2+(1-i[1][1])**2)

    if dist > tmp:

        dist = tmp

        ind.append(i)

ind
test_pred = rf_1.predict(test_data)

test_pred
#test_pid = test_df.PassengerId

submission_df = pd.DataFrame({'PassengerId': test_pid, 'Survived': test_pred}, columns=['PassengerId', 'Survived'])

submission_df.head()
submission_df.to_csv('TitanicSubmission.csv', index=False)