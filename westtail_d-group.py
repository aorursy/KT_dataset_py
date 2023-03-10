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
#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(40)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train['SalePrice'] = np.log(df_train['SalePrice'])

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

df_train = pd.get_dummies(df_train)
df_train[['GarageArea', 'WoodDeckSF', '1stFlrSF', 'FullBath' , '2ndFlrSF', 'GrLivArea', 'YearBuilt','TotalBsmtSF','OverallQual','YearRemodAdd','GarageCars']].isnull().sum()
X_train = df_train[['GarageArea', 'WoodDeckSF', '1stFlrSF', 'FullBath' , '2ndFlrSF', 'GrLivArea', 'YearBuilt','TotalBsmtSF','OverallQual','YearRemodAdd','GarageCars']]

y_train = df_train['SalePrice']



# ?????????????????????????????????????????????????????????

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(

    X_train, y_train, random_state=42)
# ????????????

# ???????????????????????????????????????

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge



# ????????????

lr = LinearRegression()

lr.fit(train_X, train_y)

print("????????????:{}".format(lr.score(test_X, test_y)))



# ???????????????

lasso = Lasso()

lasso.fit(train_X, train_y)

print("???????????????:{}".format(lasso.score(test_X, test_y)))



# ???????????????

ridge = Ridge()

ridge.fit(train_X, train_y)

print("???????????????:{}".format(ridge.score(test_X, test_y)))
df_test[['GarageArea', 'WoodDeckSF', '1stFlrSF', 'FullBath' , '2ndFlrSF', 'GrLivArea', 'YearBuilt','TotalBsmtSF','OverallQual','YearRemodAdd','GarageCars']].isnull().sum()
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)

df_test['GarageArea'] = df_test['GarageArea'].fillna(0)

df_test['GarageCars'] = df_test['GarageCars'].fillna(0)
# ID???????????????

df_test_index = df_test['Id']



# ??????????????????

df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

# ?????????????????????????????????

df_test = pd.get_dummies(df_test)

# ??????????????????????????????

df_test[df_test['TotalBsmtSF'].isnull()] 



X_test = df_test[['GarageArea', 'WoodDeckSF', '1stFlrSF', 'FullBath' , '2ndFlrSF', 'GrLivArea', 'YearBuilt','TotalBsmtSF','OverallQual','YearRemodAdd','GarageCars']]
df_test[['GarageArea', 'WoodDeckSF', '1stFlrSF', 'FullBath' , '2ndFlrSF', 'GrLivArea', 'YearBuilt','TotalBsmtSF','OverallQual','YearRemodAdd','GarageCars']].isnull().sum()
# ????????????

# ?????????

pred_y = lr.predict(X_test)

# ??????????????????????????????

submission = pd.DataFrame({'Id': df_test_index,'SalePrice': np.exp(pred_y)})

# CSV?????????????????????

submission.to_csv('submission_lr.csv', index=False)
# ???????????????

# ?????????

pred_y = lasso.predict(X_test)

# ??????????????????????????????

submission = pd.DataFrame({'Id': df_test_index,

                          'SalePrice': np.exp(pred_y)})

# CSV?????????????????????

submission.to_csv('submission_lasso.csv', index=False)
# ???????????????

# ?????????

pred_y = ridge.predict(X_test)

# ??????????????????????????????

submission = pd.DataFrame({'Id': df_test_index,

                          'SalePrice': np.exp(pred_y)})

# CSV?????????????????????

submission.to_csv('submission_ridge.csv', index=False)