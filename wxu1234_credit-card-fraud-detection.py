# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
creditcard = pd.read_csv("../input/creditcard.csv")

creditcard.head()
creditcard_X = creditcard.loc[:, (creditcard.columns != 'Class') & (creditcard.columns != 'Time')]

creditcard_X.head()

creditcard_y = creditcard['Class']

creditcard_y.head()



creditcard_fraud = creditcard.loc[creditcard['Class'] == 1]

creditcard_nonfraud = creditcard.loc[creditcard['Class'] == 0]



fraud_X = creditcard_fraud.loc[:, (creditcard_fraud.columns != 'Class') & (creditcard_fraud.columns != 'Time')]

nonfraud_X = creditcard_nonfraud.loc[:, (creditcard_nonfraud.columns != 'Class') & (creditcard_fraud.columns != 'Time')]

fraud_y = creditcard_fraud['Class']

nonfraud_y = creditcard_nonfraud['Class']
from sklearn.model_selection import train_test_split



X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(fraud_X, fraud_y, test_size = 0.1)

X_nonfraud_train, X_nonfraud_test, y_nonfraud_train, y_nonfraud_test = train_test_split(nonfraud_X, nonfraud_y, train_size = X_fraud_train.shape[0])



X_train = pd.concat([X_fraud_train, X_nonfraud_train])

y_train = pd.concat([y_fraud_train, y_nonfraud_train])

X_test = pd.concat([X_fraud_test, X_nonfraud_test])

y_test = pd.concat([y_fraud_test, y_nonfraud_test])
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



clf_pipe = Pipeline([('pca', PCA()),

               ('clf', SVC())])

param_grid = {'pca__n_components': [1,2,3],

             'clf__C': [1,1e1,1e2],

             'clf__kernel' : ['linear', 'rbf']}

clf_gs = GridSearchCV(clf_pipe, param_grid, n_jobs = -1)

clf_gs.fit(X_train, y_train)
clf_gs.best_params_

from sklearn import metrics

y_predict = clf_gs.predict(X_test)

print(metrics.classification_report(y_test, y_predict))

print(metrics.confusion_matrix(y_test, y_predict))