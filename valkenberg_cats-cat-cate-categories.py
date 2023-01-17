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


import pandas as pd

import numpy as np
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
%%time



# One Hot Encode

traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_ohe,target, test_size=0.20,random_state=101)

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000)  # MODEL



clf.fit(X_train, y_train)



pred=clf.predict_proba(X_test)[:,1]
from sklearn.model_selection import cross_validate



score=cross_validate(clf, X_train, y_train, cv=10, scoring="roc_auc")["test_score"].mean()

print(f"{score:.6f}")
pred=clf.predict_proba(test_ohe)[:,1]
df = pd.DataFrame({"id": test_id, "target": pred})

df.to_csv("submission.csv", index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')
# from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier( n_estimators = 100 )

# clf.fit(X_train,y_train)

# # score=cross_validate(clf, X_train, y_train, cv=2, scoring="roc_auc")["test_score"].mean()

# # print(f"{score:.6f}")

# probs = clf.predict_proba(test_ohe)[:,1]
# df = pd.DataFrame({"id": test_id, "target": probs})

# df.to_csv("submission.csv", index=False)
# score=cross_validate(clf, X_train, y_train, cv=5, scoring="roc_auc")["test_score"].mean()

# print(f"{score:.6f}")
# from IPython.display import FileLink

# FileLink(r'submission.csv')