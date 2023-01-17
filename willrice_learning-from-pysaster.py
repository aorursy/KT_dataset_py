#TODO: Add additional hyperparameters
import numpy as np

import pandas as pd



#import

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train['Family'] = train['SibSp'] + train['Parch']

train['Alone'] = (train['SibSp'] + train['Parch']) == 0

train['Age'] = train["Age"].fillna(train["Age"].mean())

train['Fare'] = train["Fare"].fillna(train["Fare"].mean())



test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

test['Age'] = test["Age"].fillna(test["Age"].mean())

test['Family'] = test['SibSp'] + test['Parch']

test['Alone'] = (test['SibSp'] + test['Parch']) == 0
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

train['Sex'] = lb.fit_transform(train['Sex'])

test['Sex'] = lb.fit_transform(test['Sex'])
X_train = np.array(train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Sex', 'Alone', 'Family']])

y_train = np.array(train[['Survived']]).reshape(-1)

X_test = np.array(test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Sex', 'Alone', 'Family']])
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold



cv = StratifiedKFold(n_splits=5, 

                           shuffle=True,

                           random_state=42)



#params to test

c_param = [1, 2, 5, 10, 20]



#dict of test parameters

param_grid = dict(C=c_param)



#Scikit SVC 

svc = SVC()



gs = GridSearchCV(svc, 

                  param_grid=param_grid,

                  scoring='accuracy',

                  verbose=1,

                  cv=cv)



gs.fit(X_train, y_train)



print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
y_predict = gs.predict(X_test)

submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],

                       'Survived': y_predict.T})

submit.to_csv("submit.csv", index=False)