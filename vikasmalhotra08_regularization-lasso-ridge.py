# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Lasso, LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

df.shape
df.info()
# Let's separate out the data set into train and validation data set



X_train, X_val, y_train, y_val = train_test_split(df.drop(labels=['TARGET'], axis=1), df['TARGET'], test_size=0.25, random_state=0)



X_train.fillna(0, inplace=True)

X_val.fillna(0, inplace=True)



# shapes

X_train.shape, X_val.shape
# Lets scale the data so that is ready to be used for Lasso regularization

scaler = StandardScaler()

scaler.fit(X_train)
sel = SelectFromModel(LogisticRegression(C=1, penalty = 'l1'))

sel.fit(scaler.transform(X_train), y_train)
print('Total features', X_train.shape[1])

print('Selected features', sum(sel.get_support()))

print('Removed features', np.sum(sel.estimator_.coef_ == 0))
# let's create a function to build random forests and compare the performance in training and test set:



def RandomForest(X_train, X_val, y_train, y_val):

    rf = RandomForestClassifier(n_estimators = 200, random_state = 1, max_depth = 4)

    rf.fit(X_train, y_train)

    print("Training set")

    

    pred = rf.predict_proba(X_train)

    print("Random forest roc-auc: {}".format(roc_auc_score(y_train, pred[:,1])))

    

    print("Validation set")

    pred = rf.predict_proba(X_val)

    print("Random forest roc-auc: {}".format(roc_auc_score(y_val, pred[:,1])))
# Transforming the training set and test set.

X_train_lasso = sel.transform(X_train)

X_val_lasso   = sel.transform(X_val)



RandomForest(X_train_lasso, X_val_lasso, y_train, y_val)
sfm = SelectFromModel(LogisticRegression(C=1, penalty = 'l2'))

sfm.fit(scaler.transform(X_train), y_train)
print('Total features-->',X_train.shape[1])

print('Selected featurs-->',sum(sfm.get_support()))

print('Removed featurs-->',np.sum(sfm.estimator_.coef_==0))
np.sum(np.abs(sfm.estimator_.coef_) > np.abs(sfm.estimator_.coef_).mean())
# transforming the training set and test set

X_train_l2 = sfm.transform(X_train)

X_val_l2   = sfm.transform(X_val)



RandomForest(X_train_l2, X_val_l2, y_train, y_val)