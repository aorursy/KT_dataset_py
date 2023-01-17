# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
print('Train data shape: ', train.shape)

print('Test data shape: ', test.shape)
train.columns
sns.heatmap(train.isnull())
df_train = train.ix[:,0:50]

df_train.isnull().sum()
df_train = train.ix[:,39:81]

df_train.isnull().sum()
train.ix[:,0:40].dtypes
train.ix[:,39:81].dtypes
df0 = train.drop(columns=['Alley','PoolQC','Fence','MiscFeature','Id'], axis=1)

df0['LotFrontage']=df0['LotFrontage'].fillna(df0['LotFrontage'].mean())

df0['Electrical']=df0['Electrical'].fillna(df0['Electrical'].mode()[0])

df0['BsmtQual']=df0['BsmtQual'].fillna(df0['BsmtQual'].mode()[0])

df0['BsmtCond']=df0['BsmtCond'].fillna(df0['BsmtCond'].mode()[0])

df0['BsmtExposure']=df0['BsmtExposure'].fillna(df0['BsmtExposure'].mode()[0])

df0['BsmtFinType1']=df0['BsmtFinType1'].fillna(df0['BsmtFinType1'].mode()[0])

df0['BsmtFinType2']=df0['BsmtFinType2'].fillna(df0['BsmtFinType2'].mode()[0])

df0['FireplaceQu']=df0['FireplaceQu'].fillna(df0['FireplaceQu'].mode()[0])

df0['GarageType']=df0['GarageType'].fillna(df0['GarageType'].mode()[0])

df0['GarageYrBlt']=df0['GarageYrBlt'].fillna(df0['GarageYrBlt'].mean())

df0['GarageFinish']=df0['GarageFinish'].fillna(df0['GarageFinish'].mode()[0])

df0['GarageQual']=df0['GarageQual'].fillna(df0['GarageQual'].mode()[0])

df0['GarageCond']=df0['GarageCond'].fillna(df0['GarageCond'].mode()[0])

df0['MasVnrType']=df0['MasVnrType'].fillna(df0['MasVnrType'].mode()[0])

df0['MasVnrArea']=df0['MasVnrArea'].fillna(df0['MasVnrArea'].mean())



df0.shape
sns.heatmap(df0.isnull())
df0.shape
test.head()
df_test=test.ix[:,0:40].isnull().sum()

df_test
test.ix[:,39:81].isnull().sum()
df1 = test.drop(columns=['Alley','PoolQC','Fence','MiscFeature','Id'], axis=1)

df1['MSZoning']=df1['MSZoning'].fillna(df1['MSZoning'].mode()[0])

df1['LotFrontage']=df1['LotFrontage'].fillna(df1['LotFrontage'].mean())

df1['Utilities']=df1['Utilities'].fillna(df1['Utilities'].mode()[0])

df1['Exterior1st']=df1['Exterior1st'].fillna(df1['Exterior1st'].mode()[0])

df1['Exterior2nd']=df1['Exterior2nd'].fillna(df1['Exterior2nd'].mode()[0])

df1['MasVnrType']=df1['MasVnrType'].fillna(df1['MasVnrType'].mode()[0])

df1['MasVnrArea']=df1['MasVnrArea'].fillna(df1['MasVnrArea'].mean())

df1['GarageQual']=df1['GarageQual'].fillna(df1['GarageQual'].mode()[0])

df1['GarageCond']=df1['GarageCond'].fillna(df1['GarageCond'].mode()[0])

df1['BsmtExposure']=df1['BsmtExposure'].fillna(df1['BsmtExposure'].mode()[0])

df1['BsmtFinType1']=df1['BsmtFinType1'].fillna(df1['BsmtFinType1'].mode()[0])

df1['BsmtFinSF1']=df1['BsmtFinSF1'].fillna(df1['BsmtFinSF1'].mean())

df1['BsmtFinType2']=df1['BsmtFinType2'].fillna(df1['BsmtFinType2'].mode()[0])

df1['BsmtUnfSF']=df1['BsmtUnfSF'].fillna(df1['BsmtUnfSF'].mean())

df1['TotalBsmtSF']=df1['TotalBsmtSF'].fillna(df1['TotalBsmtSF'].mean())

df1['BsmtFullBath']=df1['BsmtFullBath'].fillna(df1['BsmtFullBath'].mean())

df1['BsmtHalfBath']=df1['BsmtHalfBath'].fillna(df1['BsmtHalfBath'].mean())

df1['FireplaceQu']=df1['FireplaceQu'].fillna(df1['FireplaceQu'].mode()[0])

df1['GarageType']=df1['GarageType'].fillna(df1['GarageType'].mode()[0])

df1['GarageYrBlt']=df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].mean())

df1['GarageFinish']=df1['GarageFinish'].fillna(df1['GarageFinish'].mode()[0])

df1['GarageCars']=df1['GarageCars'].fillna(df1['GarageCars'].mean())

df1['GarageArea']=df1['GarageArea'].fillna(df1['GarageArea'].mean())

df1['GarageQual']=df1['GarageQual'].fillna(df1['GarageQual'].mode()[0])

df1['GarageCond']=df1['GarageCond'].fillna(df1['GarageCond'].mode()[0])

df1['SaleType']=df1['SaleType'].fillna(df1['SaleType'].mode()[0])

df1['BsmtCond']=df1['BsmtCond'].fillna(df1['BsmtCond'].mode()[0])

df1['BsmtQual']=df1['BsmtQual'].fillna(df1['BsmtQual'].mode()[0])
sns.heatmap(df1.isnull())
df0.shape
df1.shape
trn1 = pd.get_dummies(df0)

tst1 = pd.get_dummies(df1)
trn = trn1.drop(columns=['SalePrice'])

tst = trn1['SalePrice']

#trn = pd.get_dummies(trn1)

#trn.drop_duplicates(keep='first',inplace=True)
#trn['BsmtFinSF2']=trn['BsmtFinSF2'].fillna(trn['BsmtFinSF2'].mean())
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



x_train,x_test,y_train,y_test = train_test_split(trn,tst, test_size=0.20, random_state=23)



lr = LogisticRegression()

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
from sklearn import metrics



print("Mean Absolute Error:  ", metrics.mean_absolute_error(y_test,y_pred))

print("Mean Squared Error:  ", metrics.mean_squared_error(y_test,y_pred))

print("Root Mean Absolute Error:  ", np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
from sklearn.metrics import classification_report



print(classification_report(y_test,y_pred))
final_test=pd.get_dummies(df1)
final_test.shape
final_train = pd.get_dummies(df0)
final_train.shape