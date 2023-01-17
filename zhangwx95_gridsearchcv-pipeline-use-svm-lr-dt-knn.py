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

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train['Cabin'].fillna('U0',inplace = True)

test['Cabin'].fillna('U0',inplace = True)



train = train.drop(['Ticket','PassengerId'],axis=1)

test = test.drop(['Ticket','PassengerId'],axis=1)

combine = [train,test]
for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)

pd.crosstab(train['Title'],train['Sex'])





for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train = train.drop(['Name'],axis=1)

test = test.drop(['Name'],axis=1)

combine = [train,test]



train.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

train.head()
guess_age = np.zeros((2,3))



for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            guess = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()

            age_guess = guess.median()

            guess_age[i,j] = int(age_guess/0.5 + 0.5)*0.5

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_age[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

train.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    

for dataset in combine:

    dataset['HasCabin'] = 1

    dataset.loc[dataset['Cabin'] == 'U0', 'HasCabin'] = 0



train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train,test]
freq_port = train.Embarked.mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)

train = pd.get_dummies(train)

test = pd.get_dummies(test)

combine = [train,test]
train
train_x = train.drop('Survived', axis=1)

train_y = train['Survived']

test_x = test
from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.30, random_state = 1)
classifiers = [

    SVC(random_state = 1, kernel = 'rbf'),    

    LogisticRegression(random_state = 1),

    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),

    KNeighborsClassifier(metric = 'minkowski'),

]
classifier_names = [

            'svc', 

            'logisticregression',

            'decisiontreeclassifier',

            'kneighborsclassifier',

]
classifier_param_grid = [

            {'svc__C':[1,2], 'svc__gamma':[0.1,0.01]},

            {'logisticregression__max_iter':[50,75,100,125]},

            {'decisiontreeclassifier__max_depth':[6,9,11]},

            {'kneighborsclassifier__n_neighbors':[4,6,8]},

]
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):

    response = {}

    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)

    # 寻找最优的参数 和最优的准确率分数

    search = gridsearch.fit(train_x, train_y)

    print("GridSearch最优参数：", search.best_params_)

    print("GridSearch最优分数： %0.4lf" %search.best_score_)

    predict_y = gridsearch.predict(test_x)

    print("准确率 %0.4lf" %accuracy_score(test_y, predict_y))

    response['predict_y'] = predict_y

    response['accuracy_score'] = accuracy_score(test_y,predict_y)

    return response
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

    pipeline = Pipeline([

            ('scaler', StandardScaler()),

            (model_name, model)

    ])

    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')