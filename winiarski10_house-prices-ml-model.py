import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
test = pd.read_csv("../input/test.csv")
combine = [train_df, test_df]
train_df.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

train_df['YearBand'] = pd.qcut(train_df['YearBuilt'], q=5, labels=['0','1','2','3','4'])
test_df['YearBand'] = pd.qcut(test_df['YearBuilt'], q=5, labels=['0','1','2','3','4'])
train_df[['YearBand', 'SalePrice']].groupby(['YearBand'], as_index=False).mean().sort_values(by='YearBand', ascending=True)

train_df['BsmtBand'] = pd.qcut(train_df['BsmtFinSF1'], q=3, labels=['0','1','2'])
test_df['BsmtBand'] = pd.qcut(test_df['BsmtFinSF1'], q=3, labels=['0','1','2'])
train_df[['BsmtBand', 'SalePrice']].groupby(['BsmtBand'], as_index=False).mean().sort_values(by='BsmtBand', ascending=True)

train_df['TotalBsmtBand'] = pd.qcut(train_df['TotalBsmtSF'], q=5, labels=['0','1','2','3','4'])
test_df['TotalBsmtBand'] = pd.qcut(test_df['TotalBsmtSF'], q=5, labels=['0','1','2','3','4'])
train_df[['TotalBsmtBand', 'SalePrice']].groupby(['TotalBsmtBand'], as_index=False).mean().sort_values(by='TotalBsmtBand', ascending=True)

train_df['1stFlrBand'] = pd.qcut(train_df['1stFlrSF'], q=5, labels=['0','1','2','3','4'])
test_df['1stFlrBand'] = pd.qcut(test_df['1stFlrSF'], q=5, labels=['0','1','2','3','4'])
train_df[['1stFlrBand', 'SalePrice']].groupby(['1stFlrBand'], as_index=False).mean().sort_values(by='1stFlrBand', ascending=True)

train_df['GrLivBand'] = pd.qcut(train_df['GrLivArea'], q=5, labels=['0','1','2','3','4'])
test_df['GrLivBand'] = pd.qcut(test_df['GrLivArea'], q=5, labels=['0','1','2','3','4'])
train_df[['GrLivBand', 'SalePrice']].groupby(['GrLivBand'], as_index=False).mean().sort_values(by='GrLivBand', ascending=True)

train_df['GarageBand'] = pd.qcut(train_df['GarageArea'], q=5, labels=['0','1','2','3','4'])
test_df['GarageBand'] = pd.qcut(test_df['GarageArea'], q=5, labels=['0','1','2','3','4'])
train_df[['GarageBand', 'SalePrice']].groupby(['GarageBand'], as_index=False).mean().sort_values(by='GarageBand', ascending=True)
print(train_df.shape, test_df.shape)

train_df = train_df.drop(['Id', 'Street', 'Alley', 'Fence', 'FireplaceQu', 'PoolQC', 'MiscFeature', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 
                          'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 
                          'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'GrLivArea', 'GarageArea', 'LotArea'], axis=1)
test_df = test_df.drop(['Id', 'Street', 'Alley', 'Fence', 'FireplaceQu', 'PoolQC', 'MiscFeature', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 
                          'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 
                          'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'GrLivArea', 'GarageArea', 'LotArea'], axis=1)

print(train_df.shape, test_df.shape)
for dataset in combine:
    dataset['LotConfig'] = dataset['LotConfig'].map({"Corner":0, "FR2":0, "Inside":0, "CulDSac":1, "FR3":1})

train_df['OverallCond'] = train_df['OverallCond'].apply(lambda x: 0 if x < 5 else 1)
test_df['OverallCond'] = test_df['OverallCond'].apply(lambda x: 0 if x < 5 else 1)

# Could use the Imputer Function 
train_df = train_df.fillna({'GarageType':'Attchd', 'BsmtQual':'TA', 'MasVnrType': 'None', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'Unf', 'BsmtFinType2': 'Unf', 'Electrical': 'SBrkr',
                           'GarageFinish': 'Unf', 'GarageQual': 'TA', 'GarageCond': 'TA'})
train_df.index
ax1 = sns.stripplot(x=train_df['SaleCondition'],
                      y=train_df['SalePrice'],
                      c='DarkBlue', 
                      jitter=True)
obj_df = train_df.select_dtypes(include=['object']).copy()
obj_df.head()
X = train_df.drop(['SalePrice'], axis=1)
Y = train_df.SalePrice
one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(test_df)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='inner', 
                                                                    axis=1)
print(final_train.shape)
print(final_test.shape)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

train_X, test_X, train_y, test_y = train_test_split(final_train, Y, test_size=0.2)

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_predict = forest_model.predict(test_X)

my_model = XGBRegressor(n_estimators = 10000, learning_rate=0.01)
my_model.fit(train_X, train_y, early_stopping_rounds=50, eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)

print("Forest Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(forest_predict, test_y))))
print("XGB Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(predictions, test_y))))
predicted_prices = my_model.predict(final_test)
print(predicted_prices)
my_submission = pd.DataFrame({"Id":test.Id, 'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv', index=False)