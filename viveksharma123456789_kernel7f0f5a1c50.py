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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
test.shape
train.shape
#Data wrangling of train data
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])

train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train.drop(['PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt'], axis = 1, inplace = True)

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])

sns.heatmap(train.isnull(), yticklabels = False, cbar = False)
train.shape
train.drop(['Alley', 'Id'], axis = 1, inplace = True)
sns.heatmap(train.isnull(), yticklabels = False, cbar = False)
train.dropna(inplace = True)
train.shape
train.columns
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 

          'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 

          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofMatl', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 

          'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 

          'GarageQual', 'GarageCond', 'PavedDrive']
len(columns)
def category_onehot_multcols(multcolumns):

    train_final = final_data

    i = 0

    for fields in multcolumns:

        

        print(fields)

        train1 = pd.get_dummies(final_data[fields], drop_first = True)

        final_data.drop([fields], axis = 1, inplace = True)

        if i == 0:

            train_final = train1.copy()

        else:

            train_final = pd.concat([train_final, train1], axis = 1)

        i = i+1

        

    train_final = pd.concat([final_data, train_final], axis = 1)

    return train_final
main_train = train.copy()
#Data wrangling of test data
test.isnull().sum()
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.columns
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

test.shape
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])

test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])
test.drop(['GarageYrBlt'], axis = 1, inplace = True)
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])

test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])

test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])

test.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
test.shape
sns.heatmap(test.isnull(), yticklabels = False, cbar = False)
test.drop(['Alley'], axis = 1, inplace = True)
test.shape
final_data = pd.concat([train, test], axis = 0)
final_data.shape
final_data = category_onehot_multcols(columns)
final_data = final_data.loc[:,~final_data.columns.duplicated()]
final_data.shape
final_data.isnull().sum()
df_train = final_data.iloc[:1422,:]

df_test = final_data.iloc[1422:,:]
df_test.drop(['SalePrice'], axis = 1, inplace = True)
df_test.shape
df_train.dropna(inplace = True)
x_train = df_train.drop(['SalePrice'], axis = 1)

y_train = df_train['SalePrice']
from sklearn.metrics import accuracy_score, confusion_matrix 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 324)
price_classifier = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)

price_classifier.fit(x_train, y_train)
y_predicted = price_classifier.predict(x_test)
accuracy_score(y_test, y_predicted)*100
y_predicted
import xgboost as xg
classifier = xg.XGBRegressor()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(df_test)
y_pred
pred = pd.DataFrame(y_pred)

sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets = pd.concat([sub_df['Id'], pred], axis = 1)

datasets.columns = ['Id', 'SalePrice']

datasets.to_csv('sample_submission_copy.csv', index = False)