import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
%matplotlib inline

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# drop outliers
train_data = train_data.drop(train_data[train_data.LotArea > 200000].index)
train_data = train_data.drop(train_data[train_data.TotalBsmtSF > 6000].index)
train_data = train_data.drop(train_data[train_data.GrLivArea > 4500].index)
train_data = train_data.drop(train_data[train_data.TotalBsmtSF > train_data.GrLivArea].index)


# combine two dataset due to NaN in test data
data = pd.concat([train_data, test_data], axis = 0, sort = True)
data.reset_index(drop = True, inplace = True)
test_index = data[data.SalePrice.isnull()].index
train_index = list(set(data.index).difference(set(test_index)))

# check missing data
nan_rows = data.isnull().sum()
nan_counts = nan_rows[nan_rows > 0].sort_values(ascending = False)
nan_count_ratio = round((nan_counts / len(data)) * 100, 2)
print('Missing Value:')
print(pd.DataFrame({'count': nan_counts, 'ratio': nan_count_ratio}, index = nan_counts.index))
# drop columns that missing 50% up data
missing_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
data = data.drop(missing_columns, axis=1)

# check if BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF = TotalBsmtSF
SumBsmtSF = data.BsmtFinSF1.add(data.BsmtFinSF2).add(data.BsmtUnfSF)
diff = data.TotalBsmtSF.subtract(SumBsmtSF)
print('TotalBsmtSF - SumBsmtSF:')
print(diff.value_counts())

# check if 1stFlrSF + 2ndFlrSF + LowQualFinSF = GrLivArea
SumFlrSF = data['1stFlrSF'].add(data['2ndFlrSF']).add(data.LowQualFinSF)
diff = data.GrLivArea.subtract(SumFlrSF)
print('GrLivArea - SumFlrSF:')
print(diff.value_counts())

# drop BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF
redundent_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
data = data.drop(redundent_cols, axis=1)
# fill in nan
cato_cols = [
    'MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1',  'BsmtQual', 'BsmtCond'
]
for col in cato_cols:
    data[col] = data[col].fillna('None')

nssry_cato_cols = [
    'MSZoning', 'Utilities', 'Functional', 'Electrical',
    'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual'
]
for col in nssry_cato_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data.LotFrontage = data.LotFrontage.fillna(data.LotFrontage.mean())

num_cols = ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'GarageCars', 'GarageArea']
for col in num_cols:
    data[col] = data[col].fillna(0)
# transform year data
year_cols = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']
for col in year_cols:
    data[col] = data[col].fillna(0).map(lambda y: str(int(y)))
# data visualization for numerical data
num_cols = [
    'LotArea', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    '3SsnPorch', 'ScreenPorch', 'MiscVal', 'LotFrontage', 'MasVnrArea', 'EnclosedPorch',
]

for col in num_cols:
    train_data.plot.scatter(x=col, y='SalePrice')
import matplotlib.cm as cm
price = np.arange(9)
price = price*100000

colors = cm.rainbow(np.linspace(0, 1, len(price)))
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y.TotalBsmtSF = np.log(y.TotalBsmtSF)
    y.GrLivArea = np.log(y.GrLivArea)
    plt.scatter(y.TotalBsmtSF, y.GrLivArea, color = c)
    plt.xlabel("TotalBsmtSF")
    plt.ylabel("GrLivArea")
plt.show()
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y.GarageArea = np.log(y.GarageArea)
    y.GrLivArea = np.log(y.GrLivArea)
    plt.scatter(y.GarageArea, y.GrLivArea, color = c)
    plt.xlabel("GarageArea")
    plt.ylabel("GrLivArea")
plt.show()
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y.WoodDeckSF = np.log(y.WoodDeckSF)
    y.GrLivArea = np.log(y.GrLivArea)
    plt.scatter(y.WoodDeckSF, y.GrLivArea, color = c)
    plt.xlabel("WoodDeckSF")
    plt.ylabel("GrLivArea")
plt.show()
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y['Wood_Garage'] = np.log(y.WoodDeckSF * y.GarageArea)
    y.GrLivArea = np.log(y.GrLivArea)
    plt.scatter(y.Wood_Garage, y.GrLivArea, color = c)
    plt.xlabel("Wood*Garage")
    plt.ylabel("GrLivArea")
plt.show()
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y['YearBuilt'] = np.log(y.YearBuilt)
    y.YrSold = np.log(y.YrSold)
    plt.scatter(y.YearBuilt, y.YrSold, color = c)
    plt.xlabel("YearBuilt")
    plt.ylabel("YrSold")
plt.show()
for i , c in zip(range(0,8), colors):
    y = train_data[train_data.SalePrice > price[i]]
    y = y[y.SalePrice < price[i+1]]
    y['YearBuilt'] = y.YearBuilt
    y.YrSold = y.YrSold
    plt.scatter(y.YearBuilt, y.YrSold, color = c)
    plt.xlabel("YearBuilt")
    plt.ylabel("YrSold")
# split training data and testing data
train_data = data.iloc[train_index]
test_data = data.iloc[test_index]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
print(le)
#le.fit(["paris", "tokyo", "amsterdam"])

#le.transform([1, 2, 6]) 
# using LinearRegression
from sklearn import linear_model

train_X = train_data.loc[:, train_data.columns != 'SalePrice']
train_y = train_data.SalePrice
test_X = test_data.loc[:, train_data.columns != 'SalePrice']


train_X = train_X.ffill().bfill()
test_X = test_X.ffill().bfill()

train_X.GarageArea.replace(0, 1)
train_X.GrLivArea.replace(0, 1)
train_X.TotalBsmtSF.replace(0, 1)
test_X.GarageArea.replace(0, 1)
test_X.GrLivArea.replace(0, 1)
test_X.TotalBsmtSF.replace(0, 1)

train_X.GarageArea = np.log(train_X.GarageArea)
train_X.GrLivArea = np.log(train_X.GrLivArea)
train_X.TotalBsmtSF= np.log(train_X.TotalBsmtSF)

test_X.GarageArea = np.log(test_X.GarageArea)
test_X.GrLivArea = np.log(test_X.GrLivArea)
test_X.TotalBsmtSF= np.log(test_X.TotalBsmtSF)

small_train_x = pd.DataFrame()
small_test_x = pd.DataFrame()

small_train_x['GarageArea'] = train_X.GarageArea
small_train_x['GrLivArea'] = train_X.GrLivArea
small_train_x['TotalBsmtSF'] = train_X.TotalBsmtSF
small_train_x['YearBuilt'] = train_X.YearBuilt
small_test_x['GarageArea'] = test_X.GarageArea
small_test_x['GrLivArea'] = test_X.GrLivArea
small_test_x['TotalBsmtSF'] = test_X.TotalBsmtSF
small_test_x['YearBuilt'] = test_X.YearBuilt

small_train_x = small_train_x.replace([np.inf, -np.inf], -10)
small_test_x = small_test_x.replace([np.inf, -np.inf], -10)

print(small_train_x.head())

print(small_train_x.isnull().any().any())
print(small_test_x.isnull().any().any())
print(small_train_x.sum())
print(small_train_x.min())

model = linear_model.LinearRegression()
model.fit(small_train_x, train_y)
answer = model.predict(small_test_x)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': answer})
output.to_csv('LinearRegression.csv', index=False)

# using RandomForestRegressor
from sklearn import ensemble

model = ensemble.RandomForestRegressor(random_state=1)
model.fit(small_train_x, train_y)
answer = model.predict(small_test_x)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': answer})
output.to_csv('randomForest.csv', index=False)