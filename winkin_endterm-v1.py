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
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") 

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") 
print(train.shape,'\n',test.shape)
train.head()
test.head()
train.Alley.nunique()
train.dropna(axis=1,inplace=True)
train.shape
train.head()
train.info()
int_col=train.loc[:, train.dtypes == np.int64].columns.tolist()
train[int_col]=train[int_col].astype(np.int32)
train.info()
col_cor=(train.select_dtypes(include=np.int32).corr()['SalePrice'][:-1]>=0.5).tolist()    
col_corr=[]

for i in range(len(col_cor)):

    if(col_cor[i]==True):

        col_corr.append(train.columns[i])
col_corr
y=train['SalePrice'].values
fig, ax = plt.subplots(3,4,figsize=(14,14))

i=0

for il in col_corr:

    plt.scatter(train[il],y,marker="*",alpha=0.6)

    i+=1

    plt.xlabel(il)

    plt.ylabel("Sale Price")

    plt.subplot(3,4,i)

    

    

    

plt.tight_layout()

plt.show()
plt.scatter(train['LotArea'].values,y)
plt.scatter(train['YearBuilt'].values,y)
import seaborn as sns

sns.distplot(train.SalePrice)
train.shape
pd.get_dummies(train)
X=pd.get_dummies(train.drop(['SalePrice','Id'],1)).values

y=train['SalePrice'].values
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split

X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_trains,y_trains)
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred, squared=False)
from sklearn import linear_model

clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3,early_stopping=True)

clf.fit(X_trains, y_trains)
y_pred=clf.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)
test.dropna(axis=1,inplace=True)
loo_sum = train.groupby('YearBuilt')['SalePrice'].transform('sum')



loo_count = train.groupby('YearBuilt')['SalePrice'].transform('count')



train['SalePrice_enc'] = (loo_sum - train['SalePrice'])/(loo_count - 1)
train.head()
train['SalePrice_enc'].astype(np.int32)
np.any(np.isnan(train['SalePrice_enc']))
train['SalePrice_enc'].fillna(train['SalePrice_enc'].mean(),inplace=True)
train['SalePrice_enc'].astype(np.float32)
X=pd.get_dummies(train.drop(['SalePrice','Id'],1)).values

y=train['SalePrice'].values
from sklearn.model_selection import train_test_split

X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
reg = LinearRegression()

reg.fit(X_trains,y_trains)
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)
train['SalePrice'].max()
import lightgbm as lgb

d_train = lgb.Dataset(X_trains, label=y_trains)

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 30,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}

clf = lgb.train(params,

                d_train,

                num_boost_round=100)
y_pred=clf.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)
test.head()
pred=clf.predict(X)
X=pd.get_dummies(train.drop(['SalePrice','Id','SalePrice_enc'],1)).values

y=train['SalePrice'].values
X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import lightgbm as lgb

d_train = lgb.Dataset(X_trains, label=y_trains)

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 30,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}

clf = lgb.train(params,

                d_train,

                num_boost_round=100)
y_pred=clf.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)
pred=clf.predict(X)
pred=pred[-1]
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": pred

    })
#submission.to_csv('predict.csv',index=False)