# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv',index_col = 0)
test = pd.read_csv('../input/test.csv',index_col = 0 )
print(train.shape)
print(test.shape)
train.head()
train_label = train.pop('SalePrice')
train_label = np.log1p(train_label)
print(train_label.head())
data_all = pd.concat((train,test),axis = 0 )
data_all.loc[data_all['MSSubClass']==150,'MSSubClass'] = 160
data_all.loc[data_all['MSSubClass']==40,'MSSubClass'] = 50
data_all['MSSubClass'] = data_all['MSSubClass'].astype(str)
print(data_all['MSSubClass'].value_counts())
pd.get_dummies(data_all['MSSubClass'],prefix='MSSubClass').head()
data_all_onehot = pd.get_dummies(data_all)
data_all_onehot.head()
data_all_onehot.isnull().sum().sort_values(ascending=False).head(15)
mean_cols = data_all_onehot.mean()
data_all_onehot = data_all_onehot.fillna(mean_cols)
data_all_onehot.isnull().sum().any()
numeric_cols = data_all_onehot.columns[data_all_onehot.dtypes != 'object']
print(numeric_cols)
numeric_col_means = data_all_onehot.loc[:, numeric_cols].mean()
numeric_col_std = data_all_onehot.loc[:, numeric_cols].std()
data_all_onehot.loc[:, numeric_cols] = (data_all_onehot.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
data_all_onehot.head()
dummy_train_df = data_all_onehot.loc[train.index]
dummy_test_df = data_all_onehot.loc[test.index]
dummy_train_df.shape, dummy_test_df.shape
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
X_train = dummy_train_df.values
y_train = np.array(train_label)
X_test = dummy_test_df.values
alphas = np.logspace(-3, 3, 100)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");
print(test_score.min)
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error")
ridge = Ridge(alpha=500)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
y_final = (y_ridge + y_rf) / 2
submission_df = pd.DataFrame(data= {'Id' : test.index, 'SalePrice': y_final})
submission_df.to_csv('housing_price.csv')
submission_df.head(10)
