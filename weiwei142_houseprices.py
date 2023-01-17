import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

%config InlineBackend.figure_format='retina'# Pretty for Retina(macOS)

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/train.csv')

data.head()
data['SalePrice'].describe()
import matplotlib.pyplot as plt

import seaborn as sns
fig, ax = plt.subplots(figsize=(20,15))

sns.heatmap(data.corr(),ax=ax,annot=True,annot_kws={'size':5,'weight':'bold'},fmt='.2f')
k = 10

cols = data.corr().nlargest(k,'SalePrice')['SalePrice'].index

sns.set(font_scale=1.25)

fig, ax = plt.subplots(figsize=(20,15))

sns.heatmap(data[cols].corr(),annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,ax=ax)
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.pairplot(data[cols], size = 2.5)

plt.show()
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import numpy as np
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x = data[cols].values

y = data['SalePrice'].values

#xps = preprocessing.StandardScaler()

#yps = preprocessing.StandardScaler()

#x = xps.fit_transform(x)

#y = yps.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#X_train = xps.fit_transform(X_train)

#X_test = xps.fit_transform(X_test)

clfs = {

        'svm':svm.SVR(), 

        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),

        'BayesianRidge':linear_model.BayesianRidge()

       }

for clf in clfs:

    try:

        clfs[clf].fit(X_train, y_train)

        y_pred = clfs[clf].predict(X_test)

        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )

    except Exception as e:

        print(clf + " Error:")

        print(str(e)) 
clf = RandomForestRegressor(n_estimators=400)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)
test = pd.read_csv('../input/test.csv')

test[cols].isnull().sum()
test['GarageCars'].fillna(1.766118, inplace=True)

test['TotalBsmtSF'].fillna(1046.117970, inplace=True)
test[cols].isnull().sum()
pred = clf.predict(test[cols].values)

#origin_pred = yps.inverse_transform(pred)

origin_pred = pd.DataFrame(pred,columns=['SalePrice'])

result = pd.concat([test['Id'], origin_pred], axis=1)
result
result.to_csv('./Predictions.csv', index=False)