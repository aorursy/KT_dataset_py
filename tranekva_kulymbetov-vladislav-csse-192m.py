# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
dataset_train
corr_matrix = dataset_train.corr()

corr_matrix
mean = corr_matrix.iloc[0].abs().mean()

drop_keys = []

for k,v in corr_matrix.iloc[0].abs().items():

    tmp = v

    if tmp < mean:

        drop_keys.append(k)



drop_keys
dataset_train.drop(drop_keys, axis = 1)
from sklearn.model_selection import train_test_split



X, y = dataset_train.iloc[:,2:], dataset_train.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify = y)
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(C=5, class_weight=None, dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='ovr', n_jobs=None, penalty='l2',

                   random_state=0, solver='sag', tol=0.0001, verbose=0,

                   warm_start=False)

logreg.fit(X_train,y_train)
y_preds_result = logreg.predict(X_test)

y_preds_result
accuracy_score(y_test, y_preds_result)
y_preds_result = logreg.predict(X)

submission = pd.DataFrame({

    "ID_code": dataset_test["ID_code"],

    "target": y_preds_result

})

submission.to_csv('submission_logreg.csv', index=False)