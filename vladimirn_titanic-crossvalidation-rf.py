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
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV



from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
N_FOLDS = 10

RANDOM_STATE = 777
train = pd.read_csv("/kaggle/input/titanic/train.csv").drop(['Name','Ticket', 'Cabin'], axis=1)

test = pd.read_csv("/kaggle/input/titanic/test.csv").drop(['Name','Ticket', 'Cabin'], axis=1)
train = train.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'object' else x, axis=0)

test = test.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'object' else x, axis=0)
train.head()
test.head()
train = pd.get_dummies(train, columns = ['Pclass','Sex','SibSp','Parch','Embarked'])

test = pd.get_dummies(test, columns = ['Pclass','Sex','SibSp','Parch','Embarked'])
train.dropna(inplace=True)
X_train, y_train = train.drop(['PassengerId','Survived'], axis=1), train['Survived']
depths = np.arange(5,10,20)

grid = {'max_depth': depths}

gridsearch = GridSearchCV(DecisionTreeClassifier(), grid, scoring='neg_log_loss', cv=N_FOLDS)
%%time

gridsearch.fit(X_train, y_train)
test
best_model = gridsearch.best_estimator_
plt.barh(np.arange(len(best_model.feature_importances_)), best_model.feature_importances_)

plt.yticks(np.arange(len(X_train.columns)),X_train.columns);
from sklearn.tree import export_graphviz



def get_tree_dot_view(clf, feature_names=None, class_names=None):

    print(export_graphviz(clf, out_file=None, filled=True, feature_names=feature_names, class_names=class_names))
get_tree_dot_view(best_model, list(X_train.columns), ['NotSurv','Surv'])
depths = np.arange(5,10,20,100)

estimators = [10,20,50,100,200,1000]

grid = {'max_depth': depths, 'n_estimators': estimators, 'n_jobs': [-1]}

gridsearch = GridSearchCV(RandomForestClassifier(), grid, scoring='neg_log_loss', cv=N_FOLDS)
%%time

gridsearch.fit(X_train, y_train)
test.info()
best_model = gridsearch.best_estimator_

y_pred = best_model.predict(test[train.drop('Survived',axis=1).columns].drop('PassengerId', axis=1))
res = pd.DataFrame()

res['PassengerId'] = test['PassengerId']

res['Survived'] = y_pred

res.to_csv('submission.csv', index=False)