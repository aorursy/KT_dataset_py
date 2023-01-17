import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
%matplotlib inline
train_raw_data = pd.read_csv('../input/train.csv')
test_raw_data = pd.read_csv('../input/test.csv')
train_raw_data.shape
test_raw_data.shape
corr_mat = train_raw_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=0.8, square=True)
k = 10
cols = corr_mat.nlargest(k, ['SalePrice'])['SalePrice'].index
cm = np.corrcoef(train_raw_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, annot_kws={'size': 10}, 
                 xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_raw_data[cols], size=2.5)
plt.show()
n_train = train_raw_data.shape[0]
n_test = test_raw_data.shape[0]
all_data = pd.concat((train_raw_data, test_raw_data), sort=True).reset_index(drop=True)
all_data.drop(['SalePrice'], inplace=True, axis=1)
all_data.shape
def missing_data_stats():
    total = all_data.isnull().sum().sort_values(ascending=False)
    percent = (all_data.isnull().sum() / all_data.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(40))
missing_data_stats()
fill_na_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']

for col in fill_na_cols:
    all_data[col] = all_data[col].fillna("None")
fill_zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in fill_zero_cols:
    all_data[col] = all_data[col].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
fill_mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in fill_mode_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
missing_data_stats()
train_data = pd.concat((all_data[:n_train], train_raw_data['SalePrice']), axis=1).reset_index(drop=True)
train_data.shape
test_data = all_data[n_train:]
test_data.shape
sale_price_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:, np.newaxis])
low_range = np.sort(sale_price_scaled, axis=0)[:10]
high_range = np.sort(sale_price_scaled, axis=0)[-10:]
print('low range of the distribution')
print(low_range)
print('high range of the distribution')
print(high_range)
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
train_data.sort_values(by = 'GrLivArea', ascending=False)[:2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)

all_data = all_data.drop(all_data[all_data['Id'] == 1299].index)
all_data = all_data.drop(all_data[all_data['Id'] == 524].index)
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
def check_normal(dist):
    sns.distplot(dist, fit=norm)
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)
check_normal(train_data['SalePrice'])
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
check_normal(train_data['SalePrice'])
check_normal(train_data['GrLivArea'])
train_data['GrLivArea'] = np.log1p(train_data['GrLivArea'])
all_data['GrLivArea'] = np.log1p(all_data['GrLivArea'])
check_normal(train_data['GrLivArea'])
check_normal(train_data['TotalBsmtSF'])
train_data['TotalBsmtSF'] = np.log1p(train_data['TotalBsmtSF'])
all_data['TotalBsmtSF'] = np.log1p(all_data['TotalBsmtSF'])
check_normal(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'])
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.scatter(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF'] > 0]['SalePrice'])