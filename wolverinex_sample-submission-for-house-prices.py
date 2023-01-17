# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# loading data

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# features

train_data.columns
# concatinating training and testing data

data = pd.concat([train_data, test_data], join='outer', sort = True)

print("Data Shape :", data.shape)

data.isnull().sum().sort_values()[-35:-1]
# removing redundant data

redundant_features = ['MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

for f in redundant_features:

    data.drop(f, axis = 1, inplace = True)
#feature - 'LotFrontage'

data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace = True)



#feature - 'MasVnrArea'

data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace = True)



#feature - 'GarageArea'

data['GarageArea'].fillna(data['GarageArea'].mean(), inplace = True)



#feature - 'TotalBsmtSF'

data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean(), inplace = True)



#feature - 'BsmtUnfSF'

data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean(), inplace = True)



#feature - 'BsmtFinSF2'

data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean(), inplace = True)



#feature - 'BsmtFinSF1'

data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean(), inplace = True)



nan_features = ['GarageFinish', 'GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageType', 'BsmtCond', 'BsmtExposure',

               'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'MSZoning', 'BsmtFullBath', 'BsmtHalfBath',

               'Utilities', 'Functional', 'Electrical', 'Exterior2nd', 'KitchenQual', 'Exterior1st', 'GarageCars',

               'SaleType']

for f in nan_features:

    data[f].fillna(data[f].value_counts().idxmax(), inplace = True)
# Handling categorical data 

categorical_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig','LandSlope',

                        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',

                        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',

                        'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish',

                        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'SaleType', 'SaleCondition']



for f in categorical_features:

    dummies = pd.get_dummies(data[f], prefix = f, drop_first = False)

    data = pd.concat([data, dummies], axis = 1)



data.drop(categorical_features, axis = 1, inplace = True)
# Splitting into train and test

train = data.iloc[0:1460, :]

test = data.iloc[1460:,:]



# saving cleaned data

train.to_csv('train_clean.csv', index = False)

test.to_csv('test_clean.csv', index = False)
# loading cleaned data

train = pd.read_csv('train_clean.csv')

test = pd.read_csv('test_clean.csv')
train.drop('Id', axis=1, inplace = True)

train_y = train.pop('SalePrice')

train_x = train.values



test.drop('SalePrice', axis = 1, inplace = True)

Submission = test.pop('Id')

test_x = test.values
# training data with RandomForestRegressor

clf = RandomForestRegressor(n_estimators = 500)

clf.fit(train_x, train_y)

print(clf.score(train_x, train_y))

test_y_pred = clf.predict(test_x).astype(int)
# Saving the Submission

Submission = pd.DataFrame(Submission)

Submission['SalePrice'] = test_y_pred



Submission.to_csv('Submission.csv', index = False)