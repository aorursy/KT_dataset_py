# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
train.info()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max() #just checking that there's no missing data missing...
train.info()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);



k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train.describe()
sns.distplot(train['SalePrice']);
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
from scipy.stats import norm

from scipy import stats

#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])

train['SalePrice'] = np.log(train['SalePrice'])
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

train['HasBsmt'] = 0 

train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

train = pd.get_dummies(train)

train.info()
y_train = train["SalePrice"]

x_train = train.drop(["SalePrice","Id"],axis=1)

x_train.describe()
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split



Xtrain, xtest, ytrain, ytest = train_test_split(x_train,y_train,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#KNneighbour

knn = KNeighborsRegressor(n_neighbors = 3 , p=1)

knn.fit(Xtrain,ytrain)

y_predict_KN = knn.predict(xtest)