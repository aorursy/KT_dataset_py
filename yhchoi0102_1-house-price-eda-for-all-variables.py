import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.shape, test.shape
train.head(3)
train.dtypes.value_counts()
# 변수 종류에 따라 컬럼 분리 (Column separation according to variable type)

Id = train['Id']
target = train['SalePrice']
features = train.drop(['Id', 'SalePrice'], axis=1)

cat_col = [col for col in features.columns if features[col].dtypes == 'object']
ordinal_col = [col for col in features.columns if features[col].dtypes == 'int64']
cont_col = [col for col in features.columns if features[col].dtypes == 'float64']
null_col = [col for col in train.columns if train[col].isna().any()]
for col in null_col:
    print("{:>15} has {:>5} null values \t{:>5.2f}%".format(col, train[col].isna().sum(), 100 * train[col].isna().sum() / len(train)))
    
print(f"{len(null_col)} out of {len(train.columns)} columns are null values")
cont_col
train[cont_col].describe()
plt.figure(figsize=(8, 6))
sns.distplot(train['LotFrontage'], kde=False)
plt.show()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())
train['LotFrontage'].isna().any()
# outlier check
ulimit = np.percentile(train.LotFrontage, 99)
outliers = train['LotFrontage'].loc[train['LotFrontage'].values > ulimit]
outliers, ulimit
train['LotFrontage'].loc[train['LotFrontage'].values > ulimit] = ulimit
plt.figure(figsize=(8, 6))
sns.distplot(train['LotFrontage'], kde=True)
plt.show()
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].median())
train['MasVnrArea'].isna().any()
plt.figure(figsize=(8, 6))
sns.distplot(train['MasVnrArea'], kde=False);
train['MasVnrArea_log'] = np.log(train['MasVnrArea'].loc[train['MasVnrArea'] > 0])
train['MasVnrArea_log'] = train['MasVnrArea_log'].fillna(0)

plt.figure(figsize=(8, 6))
sns.distplot(train['MasVnrArea_log'], kde=False);
plt.figure(figsize=(8, 6))
sns.distplot(train['GarageYrBlt'], kde=False);
# train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].median())

# plt.figure(figsize=(8, 6))
# sns.distplot(train['GarageYrBlt'], kde=False);
plt.figure(figsize=(8, 6))
sns.countplot(train['MSZoning']);
plt.figure(figsize=(8, 6))
sns.countplot(train['Street']);
train['Street'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(train['Alley']);
train['Alley'].value_counts()
train['Alley'] = train['Alley'].fillna('None')
plt.figure(figsize=(8, 6))
sns.countplot(train['LotShape']);
plt.figure(figsize=(8, 6))
sns.countplot(train['LandContour']);
train['LandContour'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(train['Utilities']);
train['Utilities'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(train['LotConfig']);
train['LotConfig'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(train['LandSlope']);
train['LandSlope'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(train['Neighborhood'])
plt.xticks(rotation=75)
plt.show()
train['Neighborhood'].unique().size
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(train['Condition1'], ax=ax[0])
sns.countplot(train['Condition2'], ax=ax[1])
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['BldgType']);
plt.figure(figsize=(8, 6))
sns.countplot(train['HouseStyle']);
plt.figure(figsize=(8, 6))
sns.countplot(train['RoofStyle']);
plt.figure(figsize=(8, 6))
sns.countplot(train['RoofStyle']);
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(train['Exterior1st'], ax=ax[0])
ax[0].set_xticklabels(train['Exterior1st'].unique(), rotation=75)

sns.countplot(train['Exterior2nd'], ax=ax[1])
ax[1].set_xticklabels(train['Exterior2nd'].unique(), rotation=75)
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(train['ExterQual'], ax=ax[0])
sns.countplot(train['ExterCond'], ax=ax[1])
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['MasVnrType']);
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['MasVnrType'].isna().any()
plt.figure(figsize=(8, 6))
sns.countplot(train['Foundation']);
basement_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

plt.figure(figsize=(15, 8))
for i, col in enumerate(basement_cat):
    plt.subplot(2, 3, i+1)
    sns.countplot(train[col])
    
plt.subplots_adjust()
plt.show()
# null value data check
# 38 based on null values column
bsmt_cat_null = train.loc[train['BsmtExposure'].isna()]
bsmt_cat_null[basement_cat]
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(train['Heating'], ax=ax[0])
sns.countplot(train['HeatingQC'], ax=ax[1])
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['CentralAir']);
plt.figure(figsize=(8, 6))
sns.countplot(train['Electrical']);
# null processing
train['Electrical'] = train['Electrical'].fillna('SBrkr')
train['Electrical'].isna().any()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(train['KitchenQual'], ax=ax[0])
sns.countplot(train['Functional'], ax=ax[1])
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['FireplaceQu']);
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
train['FireplaceQu'].isna().any()
garage_col = [col for col in train.columns if 'Garage' in col]
garage_col
garage_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

plt.figure(figsize=(15, 8))
for i, col in enumerate(garage_cat):
    plt.subplot(2, 2, i+1)
    sns.countplot(train[col])

plt.subplots_adjust()
plt.show()
# 이것도 같은 데이터에 null값이 존재하는지 확인하자
garage_cat_null = train.loc[train['GarageType'].isna()]
garage_cat_null[garage_cat]
# garage null data check
train[garage_cat_null.columns][garage_col].iloc[garage_cat_null.index]
plt.figure(figsize=(8, 6))
sns.countplot(train['PavedDrive']);
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(train['PoolQC'], ax=ax[0])
sns.countplot(train['Fence'], ax=ax[1])
plt.show()
train['PoolQC'] = train['PoolQC'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')

train[['PoolQC', 'Fence']].isna().any()
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(train['PoolQC'], ax=ax[0])
sns.countplot(train['Fence'], ax=ax[1])
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['MiscFeature']);
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['MiscFeature'].isna().any()
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(train['SaleType'], ax=ax[0])
sns.countplot(train['SaleCondition'], ax=ax[1])
plt.show()
[col for col in train.columns if train[col].isna().any()]
print(ordinal_col)
train['MSSubClass'].describe()
plt.figure(figsize=(8, 6))
sns.distplot(train['MSSubClass'], kde=True)
plt.xticks(range(0, 200, 20))
plt.show()
train['LotArea'].describe()
plt.figure(figsize=(8, 6))
sns.distplot(train['LotArea'], kde=False)
plt.show()
ulimit = np.percentile(train['LotArea'].values, 99)
train['LotArea'].loc[train['LotArea'] > ulimit] = ulimit
plt.figure(figsize=(8, 6))
sns.distplot(train['LotArea'], kde=True)
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(train['OverallQual'], ax=ax[0])
sns.countplot(train['OverallCond'], ax=ax[1])

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.distplot(train['YearBuilt'], kde=False, ax=ax[0])
ax[0].set_xticks(range(1880, 2020, 20))

sns.distplot(train['YearRemodAdd'], kde=False, ax=ax[1])
ax[1].set_xticks(range(1880, 2020, 20))
plt.show()
train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].describe()
fig, ax = plt.subplots(2, 2, figsize=(20, 15))

sns.distplot(train['BsmtFinSF1'], kde=False, ax=ax[0, 0])

sns.distplot(train['BsmtFinSF2'], kde=False, ax=ax[0, 1])

sns.distplot(train['BsmtUnfSF'], kde=True, ax=ax[1, 0])

sns.distplot(train['TotalBsmtSF'], kde=True, ax=ax[1, 1])

plt.show()
# TotalBsmtSF outlier
ulimit = np.percentile(train['TotalBsmtSF'].values, 99)
train['TotalBsmtSF'].loc[train['TotalBsmtSF'] > ulimit] = ulimit
sns.distplot(train['TotalBsmtSF'], kde=True);
train[['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea']].describe()
fig, ax = plt.subplots(2, 2, figsize=(20, 15))

sns.distplot(train['1stFlrSF'], kde=True, ax=ax[0, 0])

sns.distplot(train['2ndFlrSF'], kde=True, ax=ax[0, 1])

sns.distplot(train['LowQualFinSF'], kde=False, ax=ax[1, 0])

sns.distplot(train['GrLivArea'], kde=True, ax=ax[1, 1])

plt.show()
train["LowQualFinSF"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(train['BsmtFullBath'], ax=ax[0])
sns.countplot(train['BsmtHalfBath'], ax=ax[1])

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(train['FullBath'], ax=ax[0])
sns.countplot(train['HalfBath'], ax=ax[1])
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(16, 6))

sns.countplot(train['BedroomAbvGr'], ax=ax[0])
sns.countplot(train['KitchenAbvGr'], ax=ax[1])
sns.countplot(train['TotRmsAbvGrd'], ax=ax[2])

plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(train['Fireplaces'])
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(train['GarageCars'], color='#34495e', ax=ax[0])
sns.distplot(train['GarageArea'], kde=True, ax=ax[1])

plt.show()
plt.figure(figsize=(8, 6))
sns.distplot(train['WoodDeckSF'], kde=True)
plt.show()
fig, ax = plt.subplots(2, 2, figsize=(16, 8))

sns.distplot(train['OpenPorchSF'], kde=False, ax=ax[0, 0])

sns.distplot(train['EnclosedPorch'], kde=False, ax=ax[0, 1])

sns.distplot(train['3SsnPorch'], kde=False, ax=ax[1, 0])

sns.distplot(train['ScreenPorch'], kde=False, ax=ax[1, 1])

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.distplot(train['PoolArea'], kde=False, ax=ax[0])
sns.distplot(train['MiscVal'], kde=False, ax=ax[1])

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.countplot(train['MoSold'], color='#34495e', ax=ax[0])
sns.countplot(train['YrSold'], color='#34495e', ax=ax[1])

plt.show()
