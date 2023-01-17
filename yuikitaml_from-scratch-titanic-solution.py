# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import random
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gsub = pd.read_csv('../input/gender_submission.csv')
print(train.columns)
train.head(10)
train = train.assign(Alone = (train.SibSp == 0) & (train.Parch == 0))
def formatting(d, nf, cf, tr=[]):
    """
    nf = numerical features
    cf = categolical features
    One stop function for
    - Drop NaN
    - One-hot encoding
    """
    
    # Drop NaN
    d = d[nf + cf + tr]
    d = d.dropna(axis=0)
    
    # One-hot encoding
    num_df = d[nf]
    cat_df = pd.get_dummies(d[cf])
    X = pd.concat([num_df, cat_df], axis=1)
    if len(tr) != 0:
        y = d[tr]
    else:
        y = None
    return X, y
def fit(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred_y)
    return model, mae
X, y= formatting(train, nf=['Alone'], cf=['Sex'], tr=['Survived'])
model, mae = fit(X, y)
print(mae)
# Bit confusing...
test = test.fillna(0)
test = test.assign(Alone = ((test.SibSp == 0) & (test.Parch == 0)))
age_column = test[['Alone']]
sex_column = pd.get_dummies(test['Sex'])
test_X = pd.concat([age_column, sex_column], axis=1)
passenger_id = list(test.PassengerId)
result = model.predict(test_X)
result = [int(r) for r in result]
submission = pd.DataFrame({ 'PassengerId': passenger_id, 'Survived': result })
submission.to_csv("Submission.csv", index=False)
print(len(submission))
submission.describe()

