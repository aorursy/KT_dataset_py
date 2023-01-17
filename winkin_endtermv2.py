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
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape, test.shape
train.head()
test.head()
train.info()
test.info()
int_col=train.loc[:, train.dtypes == np.int64].columns.tolist()
train[int_col]=train[int_col].astype(np.int32)
int_col=test.loc[:, test.dtypes == np.int64].columns.tolist()
test[int_col]=test[int_col].astype(np.int32)
nan_val=(train.isnull().sum()>1000).tolist()
remove_val=[]

for i in range(len(nan_val)):

    if(nan_val[i]==True):

        remove_val.append(train.columns[i])
nan_val=(test.isnull().sum()>1000).tolist()
test_remove_val=[]

for i in range(len(nan_val)):

    if(nan_val[i]==True):

        test_remove_val.append(test.columns[i])
remove_val
test_remove_val
train.drop(columns=remove_val,inplace=True)

test.drop(columns=test_remove_val,inplace=True)
train.shape ,test.shape
train.describe()
test.describe()
int_col=train.loc[:, (train.dtypes == np.int32) | (train.dtypes==np.float64)].columns.tolist()
int_col
X=train[int_col]

y=train["SalePrice"]
X.drop(columns=['SalePrice'],inplace=True)
X.dropna(axis=1,inplace=True)
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
imp_coef = coef.sort_values()

from matplotlib import pyplot as plt

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
drop=imp_coef[imp_coef<=0].index
plt.scatter(train['YearBuilt'].values,train['SalePrice'].values)
np.percentile(train["SalePrice"].values,[1,99])
np.percentile(train["YearBuilt"].values,[1,99])
plt.scatter(train[(train['SalePrice']<442500) & (train['YearBuilt']<2009)]['YearBuilt'].values,train[(train['YearBuilt']<2009) & (train['SalePrice']<442500)]['SalePrice'].values)
train.dropna(axis=1,inplace=True)
train.drop(columns=drop,inplace=True)
train.shape
test.shape
X=pd.get_dummies(train[(train['SalePrice']>62000) & (train['SalePrice']<442500)].drop(columns=['SalePrice'])).values

y=train[(train['SalePrice']>62000) & (train['SalePrice']<442500)]['SalePrice'].values
X.shape , y.shape
from sklearn.model_selection import train_test_split

X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_trains.shape
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_trains,y_trains)
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error ,r2_score

mean_squared_error(y_test, y_pred, squared=False), r2_score(y_test,y_pred)
X_topred=pd.get_dummies(train).drop(columns=['SalePrice']).values
pred=reg.predict(X_topred)
pred=pred[-1]
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": pred

    })
submission.to_csv('predict.csv',index=False)