import numpy as np
import pandas as pd
from xgboost import XGBRegressor
data = pd.read_csv('../input/train.csv')
data.describe()
import missingno as msno

features = list(data)
msno.matrix(data.iloc[:, 0:50])
msno.matrix(data.iloc[:, 50:])
data = data.fillna(value={'LotFrontage':0, 'Alley':'None', 'BsmtQual':'None', 'BsmtCond':'None',
                   'BsmtExposure':'None', 'BsmtFinType1':'None', 'BsmtFinType2':'None',
                   'FireplaceQu': 'None', 'GarageType': 'None', 'GarageYrBlt':-1, 'GarageFinish':'None',
                   'GarageQual':'None', 'GarageCond':'None', 'PoolQC':'None', 'MiscFeature':'None',
                   'Fence':'None', 'MiscFeature':'None'})
data['LotFrontageNaN'] = (data['LotFrontage'] == 0).astype(int)
data.fillna(data.mode().iloc[0], inplace=True)
test_X = pd.read_csv('../input/test.csv')
test_X = test_X.fillna(value={'LotFrontage':0, 'Alley':'None', 'BsmtQual':'None', 'BsmtCond':'None',
                   'BsmtExposure':'None', 'BsmtFinType1':'None', 'BsmtFinType2':'None',
                   'FireplaceQu': 'None', 'GarageType': 'None', 'GarageYrBlt':-1, 'GarageFinish':'None',
                   'GarageQual':'None', 'GarageCond':'None', 'PoolQC':'None', 'MiscFeature':'None',
                   'Fence':'None', 'MiscFeature':'None'})
test_X['LotFrontageNaN'] = (test_X['LotFrontage'] == 0).astype(int)
test_X.fillna(data.mode().iloc[0], inplace=True)
categorical_variables = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
data = pd.get_dummies(data, columns=categorical_variables)
y = data['SalePrice']
X = data.drop(['Id', 'SalePrice'], axis=1)
ids = test_X['Id']
test_X = pd.get_dummies(test_X, columns=categorical_variables)
X, test_X = X.align(test_X, join='left', axis=1)
from sklearn.model_selection import train_test_split
X, y = X.values, y.values
train_X, train_y = X, y
train_y = train_y.reshape((-1, 1))
print(train_X.shape, train_y.shape)
from xgboost import XGBRegressor
model = XGBRegressor(learning_rate=0.1, n_estimators=400)
model.fit(train_X, train_y)
y_pred = model.predict(test_X.values)
output = pd.DataFrame({'Id': ids, 'SalePrice': y_pred})
output.to_csv('submission.csv', index=False)