# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression

%matplotlib inline 
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.describe
train.corr()
train
train.columns
train.SalePrice.describe()
sns.distplot(train.SalePrice)
#salepriceと相関が高い関数を探す

k = 10

corrmat = train.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea']

sns.pairplot(train[cols],size = 2.5)

plt.show()
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(index = train[train['Id'] == 1299].index)

train = train.drop(index = train[train['Id'] == 524].index)
sns.set()

cols = ['SalePrice','OverallCond']

sns.pairplot(train[cols],size = 2.5)

plt.show()
#数値の大きい上位2つ

train.sort_values(by = 'OverallCond', ascending = False)[:2]
train = train.drop(index = train[train['Id'] == 584].index)

train = train.drop(index = train[train['Id'] == 305].index)
X = train[["OverallCond"]].values

y = train["SalePrice"].values



slr = LinearRegression()



slr.fit(X,y)



print('傾き：{0}'.format(slr.coef_[0]))

print('y切片: {0}'.format(slr.intercept_))
#描画

plt.scatter(X,y)

plt.plot(X,slr.predict(X),color='red')

plt.show()
train.isnull().sum()
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
train_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_test.head()
X_test = train_test[["OverallCond"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)
y_test_pred
train_test["SalePrice"] = y_test_pred
train_test[["Id","SalePrice"]].head()

train_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
from sklearn.linear_model import LogisticRegression

cols = ["GrLivArea","OverallCond"] 

X = train[cols]

y = train['SalePrice']



model = LogisticRegression()

model.fit(X,y)



from sklearn.metrics import accuracy_score

train_predicted = model.predict(X)

accuracy_score(train_predicted, y)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
cols = ["GrLivArea","OverallCond"] 

X_test=test[cols]

print(X_test.dtypes)

test_predicted = model.predict(X_test)
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = list(map(int, test_predicted))

sub.to_csv('submission.csv', index=False)