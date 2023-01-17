import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
data = data[["NA_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]]
sns.pairplot(data = data)
data = data[["NA_Sales", "Global_Sales"]]
print('Input parameter: NA_Sales')
print('Output parameter: Global_Sales')
data.head()
if True in pd.isnull(data) :
    print('Incomplete data')
else :
    print('Incomplete data is absent')
    
data.describe()
print('Dependency between north american and global sales')
sns.pairplot(data=data)
data = data[data['NA_Sales']<data['NA_Sales'].quantile(0.99)]
sns.pairplot(data=data)
Y = data['Global_Sales']
X = data.drop('Global_Sales', axis=1)
del(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=228)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
transformer = preprocessing.Normalizer().fit(X_train)
X_train, X_test = transformer.transform(X_train), transformer.transform(X_test)

lasso = linear_model.Lasso(alpha=0.000001, normalize=True, max_iter=2200)#,fit_intercept=False
lasso.fit(X_train, Y_train)
lasso_result = lasso.predict(X_test)

train_err = 1 - lasso.score(X_train, Y_train)
test_err = 1 - lasso.score(X_test, Y_test)

print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
ridge = linear_model.Ridge(alpha=0.01,normalize=True,solver='lsqr')
ridge.fit(X_train, Y_train)
ridge_result = ridge.predict(X_test)

train_err = 1 - ridge.score(X_train, Y_train)
test_err = 1 - ridge.score(X_test, Y_test)

print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
elnet = linear_model.ElasticNet(alpha=0.00001,l1_ratio=1, normalize=True, max_iter=1500)
elnet.fit(X_train, Y_train)
elnet_result = elnet.predict(X_test)

train_err = 1 - elnet.score(X_train, Y_train)
test_err = 1 - elnet.score(X_test, Y_test)

print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))
ensemble = VotingRegressor([('Lasso', lasso), ('Ridge', ridge), ('ElNet', elnet)], weights = [0.5, 0.5, 0.5])

ensemble.fit(X_train, Y_train)
ensemble_result = ensemble.predict(X_test)

train_err = 1 - ensemble.score(X_train, Y_train)
test_err = 1 - ensemble.score(X_test, Y_test)

print('train error =', train_err, '\ntest error =', test_err, '\ndifference =', abs(train_err - test_err))