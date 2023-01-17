import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import r2_score
train = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip', compression='zip')

test = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip', compression='zip')
train.head()
X = train[[x for x in train.columns if x[0]=="X"]].copy()

test_X = test.iloc[:, 1:].copy()



drop = []

for col in X:

    if X[col].max() == 0:

        drop.append(col)



X.drop(drop, axis=1, inplace=True)

test_X.drop(drop, axis=1, inplace=True)





for col in X.iloc[:, :8].columns:

    enc = LabelEncoder()

    enc.fit(X[col])

    X[col] = enc.fit_transform(X[col])

    test_X[col] = enc.fit_transform(test_X[col])



for col in X.columns[:8]:

    dummy = pd.get_dummies(X[col], drop_first=False)

    dummy.columns = [f'{col}_{x}' for x in dummy.columns]

    X = X.merge(dummy, right_index=True, left_index=True)

    

    dummy = pd.get_dummies(test_X[col], drop_first=False)

    dummy.columns = [f'{col}_{x}' for x in dummy.columns]

    test_X = test_X.merge(dummy, right_index=True, left_index=True)

    

test_X.drop(test_X.columns[:8], axis=1, inplace=True)

X.drop(X.columns[:8], axis=1, inplace=True)



y = train.y



train_X, val_X, train_y, val_y = train_test_split(X, y)
mb_tree = RandomForestRegressor(random_state=1, n_estimators=200, max_features='log2')

mb_tree.fit(train_X, train_y)

pred_y = mb_tree.predict(val_X)

r2_score(val_y, pred_y)
test_X.drop(set(test_X.columns)-set(train_X.columns), axis=1, inplace=True)



pred_y = mb_tree.predict(test_X)
test.ID
len(pred_y)
test.ID
preds = pd.Series(pred_y)
submition = pd.DataFrame([test.ID, preds]).T

submition.columns = ['ID','y']
submition = submition.astype({'ID':int, 'y':float})
submition.to_csv("/kaggle/working/submition.csv", index=False)