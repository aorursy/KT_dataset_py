import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
from __future__ import division
from IPython.core.debugger import set_trace

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
print(os.listdir("../input"))
houses = pd.read_csv('../input/train.csv')
test_houses = pd.read_csv('../input/test.csv')
houses.shape
houses.head()
houses.info()
houses['SalePrice'].describe()
plt.figure(figsize=(8, 7))
sns.distplot(houses['SalePrice'], kde=False)
plt.show()
print(f"Skewness: {houses['SalePrice'].skew()}")
print(f"Kurtosis: {houses['SalePrice'].kurt()}")
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['MSSubClass'])
plt.subplot(122)
sns.boxplot(data=houses, x='MSSubClass', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['MSZoning'])
plt.subplot(122)
sns.boxplot(data=houses, x='MSZoning', y='SalePrice')
plt.show()
houses['LotFrontage'].describe()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(houses['LotFrontage'].dropna())
plt.subplot(122)
sns.scatterplot(x=houses['LotFrontage'], y=houses['SalePrice'], alpha=0.4)
plt.show()
houses.loc[:, ['LotFrontage', 'SalePrice']].corr()
houses['LotArea'].describe()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(houses['LotArea'])
plt.subplot(122)
sns.scatterplot(data=houses, x='LotArea', y='SalePrice', alpha=0.4)
plt.show()
houses.loc[:, ['LotArea', 'SalePrice']].corr()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['Street'])
plt.subplot(122)
sns.boxplot(x=houses['Street'], y=houses['SalePrice'])
plt.show()
df = houses.copy()
df['Alley'].fillna('None', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(data=df, x='Alley')
plt.subplot(122)
sns.boxplot(data=df, x='Alley', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 8))
plt.subplot(121)
sns.countplot(houses['LotShape'])
plt.subplot(122)
sns.boxplot(data=houses, x='LotShape', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['LandContour'])
plt.subplot(122)
sns.boxplot(data=houses, x='LandContour', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['LotConfig'])
plt.subplot(122)
sns.boxplot(data=houses, x='LotConfig', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['LandSlope'], order=['Gtl', 'Mod', 'Sev'])
plt.subplot(122)
sns.boxplot(data=houses, x='LandSlope', y='SalePrice', order=['Gtl', 'Mod', 'Sev'])
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['Neighborhood'])
plt.xticks(rotation=90)
plt.subplot(122)
sns.boxplot(data=houses, x='Neighborhood', y='SalePrice')
plt.xticks(rotation=90)
plt.show()
df = houses.copy()

#new feature indicate if house is adjacent to main road
df['nearMainRoad'] = (df['Condition1'] == 'Artery') | (df['Condition2'] == 'Artery')

#new feature indicate if houses is within 200' of Railroad
df['within200RailRoad'] = (df['Condition1'] == 'RRNn') | (df['Condition1'] == 'RRNe') | (df['Condition2'] == 'RRNn') | (df['Condition2'] == 'RRNe')

#new feature indicate if houses is within near Railroad
df['nearRailRoad'] = (df['Condition1'] == 'RRAn') | (df['Condition1'] == 'RRAe') | (df['Condition2'] == 'RRAn') | (df['Condition2'] == 'RRAe')

#new feature indicate if houses is near park
df['nearPark'] = (df['Condition1'] == 'PosN') | (df['Condition1'] == 'PosA') | (df['Condition2'] == 'PosN') | (df['Condition2'] == 'PosA')
#plot nearMainRoad
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['nearMainRoad'])
plt.subplot(122)
sns.boxplot(data=df, x='nearMainRoad', y='SalePrice')
plt.show()
#plot within200RailRoad
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['within200RailRoad'])
plt.subplot(122)
sns.boxplot(data=df, x='within200RailRoad', y='SalePrice')
plt.show()
#plot nearRailRoad
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['nearRailRoad'])
plt.subplot(122)
sns.boxplot(data=df, x='nearRailRoad', y='SalePrice')
plt.show()
#combile nearRailRoad and within200RailRoad
df['closeToRailRoad'] = df['nearRailRoad'] | df['within200RailRoad']

#plot closeToRailRoad
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['closeToRailRoad'])
plt.subplot(122)
sns.boxplot(data=df, x='closeToRailRoad', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 8))
plt.subplot(121)
sns.countplot(houses['BldgType'])
plt.subplot(122)
sns.boxplot(data=houses, x='BldgType', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 8))
plt.subplot(121)
sns.countplot(houses['HouseStyle'])
plt.subplot(122)
sns.boxplot(data=houses, x='HouseStyle', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['OverallQual'])
plt.subplot(122)
sns.boxplot(data=houses, x='OverallQual', y='SalePrice')
plt.show()
df = houses.copy()
df['OverallQualCate'] = pd.cut(df['OverallQual'], bins=[0, 3, 7, 10], labels=['low', 'medium', 'high'])

plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['OverallQualCate'])
plt.subplot(122)
sns.boxplot(data=df, x='OverallQualCate', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['OverallCond'])
plt.subplot(122)
sns.boxplot(data=houses, x='OverallCond', y='SalePrice')
plt.show()
#bin OverallCond into 3 groups
df = houses.copy()
df['OverallCondCate'] = pd.cut(df['OverallCond'], bins=[0, 3, 7, 10], labels=['low', 'medium', 'high'])

plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['OverallCondCate'])
plt.subplot(122)
sns.boxplot(data=df, x='OverallCondCate', y='SalePrice')
plt.show()
plt.figure(figsize=(8, 7))
sns.lineplot(data=houses, x='YearBuilt', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['RoofStyle'])
plt.subplot(122)
sns.boxplot(data=houses, x='RoofStyle', y='SalePrice')
plt.show()
#Exterior1st
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['Exterior1st'])
plt.xticks(rotation=90)
plt.subplot(122)
sns.boxplot(data=houses, x='Exterior1st', y='SalePrice')
plt.xticks(rotation=90)
plt.show()
#Exterior2nd
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['Exterior2nd'])
plt.xticks(rotation=90)
plt.subplot(122)
sns.boxplot(data=houses, x='Exterior2nd', y='SalePrice')
plt.xticks(rotation=90)
plt.show()
df = houses.copy()
df['MasVnrArea'].fillna('None', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['MasVnrType'])
plt.subplot(122)
sns.boxplot(data=df, x='MasVnrType', y='SalePrice')
plt.show()
df = houses.copy()
df['MasVnrArea'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['MasVnrArea'])
plt.subplot(122)
sns.scatterplot(data=df, x='MasVnrArea', y='SalePrice')
plt.show()
df.loc[:, ['MasVnrArea', 'SalePrice']].corr()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['ExterQual'], order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='ExterQual', y='SalePrice', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['ExterCond'], order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='ExterCond', y='SalePrice', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
plt.figure(figsize=(8, 7))
sns.boxplot(data=df, x='ExterQual', y='OverallQual', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['Foundation'])
plt.subplot(122)
sns.boxplot(data=df, x='Foundation', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(houses['Utilities'], order=['ELO', 'NoSeWa', 'NoSeWr', 'AllPub'])
plt.subplot(122)
sns.boxplot(data=df, x='Utilities', y='SalePrice', order=['ELO', 'NoSeWa', 'NoSeWr', 'AllPub'])
plt.show()
houses['Utilities'].unique()
houses.loc[houses['Utilities'] == 'NoSeWa', :]

df = houses.copy()
df['BsmtQual'].fillna('None', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['BsmtQual'], order=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='BsmtQual', y='SalePrice', order=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
df['BsmtCond'].fillna('None', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['BsmtCond'], order=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='BsmtCond', y='SalePrice', order=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
df['BsmtExposure'].fillna('NoBasement', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['BsmtExposure'], order=['NoBasement', 'No', 'Mn', 'Av', 'Gd'])
plt.subplot(122)
sns.boxplot(data=df, x='BsmtExposure', y='SalePrice',order=['NoBasement', 'No', 'Mn', 'Av', 'Gd'])
plt.show()
df = houses.copy()
df['BsmtFinSF1'].fillna('NoBasement', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['BsmtFinSF1'])
plt.subplot(122)
sns.scatterplot(data=df, x='BsmtFinSF1', y='SalePrice')
plt.show()
houses.loc[:, ['BsmtFinSF1', 'SalePrice']].corr()
df = houses.copy()
df['BsmtUnfSF'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['BsmtUnfSF'])
plt.subplot(122)
sns.scatterplot(data=df, x='BsmtUnfSF', y='SalePrice')
plt.show()
houses.loc[:, ['BsmtUnfSF', 'SalePrice']].corr()
df = houses.copy()
df['TotalBsmtSF'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['TotalBsmtSF'])
plt.subplot(122)
sns.scatterplot(data=df, x='TotalBsmtSF', y='SalePrice')
plt.show()
houses.loc[:, ['TotalBsmtSF', 'SalePrice']].corr()
df = houses.copy()
df['Heating'].fillna('NoHeating', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['Heating'])
plt.subplot(122)
sns.boxplot(data=df, x='Heating', y='SalePrice')
plt.show()
df = houses.copy()
df['HeatingQC'].fillna('NoHeating', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['HeatingQC'], order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='HeatingQC', y='SalePrice', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
df['CentralAir'].fillna('NA', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['CentralAir'])
plt.subplot(122)
sns.boxplot(data=df, x='CentralAir', y='SalePrice')
plt.show()
df = houses.copy()
df['Electrical'].fillna('NA', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['Electrical'])
plt.subplot(122)
sns.boxplot(data=df, x='Electrical', y='SalePrice')
plt.show()
df = houses.copy()
df['1stFlrSF'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['1stFlrSF'])
plt.subplot(122)
sns.scatterplot(data=df, x='1stFlrSF', y='SalePrice')
plt.show()
houses.loc[:, ['1stFlrSF', 'SalePrice']].corr()
df = houses.copy()
df['2ndFlrSF'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['2ndFlrSF'])
plt.subplot(122)
sns.scatterplot(data=df, x='2ndFlrSF', y='SalePrice')
plt.show()
df.loc[:, ['2ndFlrSF', 'SalePrice']].corr()
oneFlrHouses = df.loc[df['2ndFlrSF'] == 0, :]
twoFlrHouses = df.loc[df['2ndFlrSF'] > 0, :]
mean1 = oneFlrHouses['SalePrice'].mean()
mean2 = twoFlrHouses['SalePrice'].mean()

plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(oneFlrHouses['SalePrice'])
plt.gca().axvline(mean1, color='r' , linestyle='--')
plt.text(x=mean1 + 30000,y= 0.000005, s=f'Mean {mean1:.2f}', color='r')
plt.title("Houses with only 1 floor Price Distribution")
plt.subplot(122)
sns.distplot(twoFlrHouses['SalePrice'])
plt.gca().axvline(mean2, color='r', linestyle='--')
plt.text(x=mean2 + 30000,y= 0.000005, s=f'Mean {mean2:.2f}', color='r')
plt.title("Houses with 2 floors Price Distribution")
plt.show()
df = houses.copy()
df['GrLivArea'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['GrLivArea'])
plt.subplot(122)
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice')
plt.show()
df.loc[:, ['GrLivArea', 'SalePrice']].corr()
df = houses.copy()
df['AboveGrSF'] = df['1stFlrSF'] + df['2ndFlrSF']

df['GrLivArea'].fillna(0, inplace=True)
df['AboveGrSF'].fillna(0, inplace=True)


plt.figure(figsize=(8, 7))
sns.scatterplot(data=df, x='AboveGrSF', y='GrLivArea')
plt.show()
df.loc[:, ['GrLivArea', 'totalFlrSF']].corr()
df = houses.copy()
df['TotalBath'] = df['FullBath'] + df['BsmtFullBath'] + df['BsmtHalfBath'] +df['HalfBath']

plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['TotalBath'])
plt.subplot(122)
sns.boxplot(data=df, x='TotalBath', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['BedroomAbvGr'])
plt.subplot(122)
sns.boxplot(data=df, x='BedroomAbvGr', y='SalePrice')
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['KitchenAbvGr'])
plt.subplot(122)
sns.boxplot(data=df, x='KitchenAbvGr', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['KitchenQual'], order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='KitchenQual', y='SalePrice', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['TotRmsAbvGrd'])
plt.subplot(122)
sns.boxplot(data=df, x='TotRmsAbvGrd', y='SalePrice')
plt.show()
df = houses.copy()
df['FireplaceQu'].fillna('NoFirePlace', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['FireplaceQu'])
plt.subplot(122)
sns.boxplot(data=df, x='FireplaceQu', y='SalePrice')
plt.show()
df['FireplaceQu'].fillna(0, inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['FireplaceQu'], order=['NoFirePlace','Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='FireplaceQu', y='SalePrice', order=['NoFirePlace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
df['GarageType'].fillna('NoGarage', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['GarageType'])
plt.subplot(122)
sns.boxplot(data=df, x='GarageType', y='SalePrice')
plt.show()
plt.figure(figsize=(8, 7))
sns.lineplot(df['GarageYrBlt'], df['SalePrice'])
plt.show()
plt.figure(figsize=(8,7))
sns.scatterplot(houses['GarageYrBlt'], houses['YearBuilt'])
plt.show()
houses.loc[:, ['GarageYrBlt','YearBuilt']].corr()
plt.figure(figsize=(8,7))
sns.countplot(houses['GarageFinish'])
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['GarageCars'])
plt.subplot(122)
sns.boxplot(data=df, x='GarageCars', y='SalePrice')
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(df['GarageArea'])
plt.subplot(122)
sns.scatterplot(data=df, x='GarageArea', y='SalePrice')
plt.show()
houses.loc[:, ['GarageArea', 'SalePrice']].corr()
plt.figure(figsize=(8, 7))
sns.boxplot(data=df, x='GarageCars', y='GarageArea')
plt.show()
df = houses.copy()
df['GarageQual'].fillna('NoGarage', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['GarageQual'], order=['NoGarage','Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='GarageQual', y='SalePrice', order=['NoGarage','Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
df['GarageCond'].fillna('NoGarage', inplace=True)
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['GarageCond'], order=['NoGarage','Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.subplot(122)
sns.boxplot(data=df, x='GarageCond', y='SalePrice', order=['NoGarage','Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['PavedDrive'], order=['N','P', 'Y'])
plt.subplot(122)
sns.boxplot(data=df, x='PavedDrive', y='SalePrice', order=['N','P', 'Y'])
plt.show()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.distplot(houses['WoodDeckSF'])
plt.subplot(122)
sns.scatterplot(data=houses, x='WoodDeckSF', y='SalePrice')
plt.show()
houses.loc[:, ['WoodDeckSF', 'SalePrice']].corr()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.boxplot(houses['PoolArea'])
plt.subplot(122)
sns.scatterplot(data=houses, x='PoolArea', y='SalePrice')
plt.show()
df = houses.copy()

#Any houses has an Elevator?
df['hasElevator'] = df['MiscFeature'] == 'Elev'
sns.countplot(df['hasElevator'])
plt.show()
df['hasElevator'].unique()
#Any houses has 2 garages?
df['twoGarage'] = df['MiscFeature'] == 'Gar2'
sns.countplot(df['twoGarage'])
plt.show()
#Any houses has tennis court?

df['hasTennisCourt'] = df['MiscFeature'] == 'TenC'
sns.countplot(df['hasTennisCourt'])
plt.show()
df.loc[(df['hasTennisCourt'] == True), ['SalePrice']]
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['MoSold'])
plt.subplot(122)
sns.boxplot(data=df, x='MoSold', y='SalePrice')
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['YrSold'])
plt.subplot(122)
sns.boxplot(data=df, x='YrSold', y='SalePrice')
plt.show()
houses['YrSold'].unique()
df = houses.copy()
plt.figure(figsize=(16, 7))
sns.boxplot(df['YrSold'], df['YearBuilt'])
plt.show()
houses['YrSold'].shape
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['SaleType'])
plt.subplot(122)
sns.boxplot(data=df, x='SaleType', y='SalePrice')
plt.show()
df = houses.copy()
plt.figure(figsize=(16, 7))
plt.subplot(121)
sns.countplot(df['SaleCondition'])
plt.subplot(122)
sns.boxplot(data=df, x='SaleCondition', y='SalePrice')
plt.show()
#transform MSSubClass into string
houses['MSSubClass'] = houses['MSSubClass'].apply(str)
df = houses.copy()
qualFeatures = [c for c in df.columns if c.endswith('Qual') & (c != 'OverallQual')]
#df[qualFeatures].fillna('None', inplace=True)

n_col = 2
n_row = np.ceil(len(qualFeatures) / n_col)

plt.figure(figsize=(15, 10))
for index, value in enumerate(qualFeatures):
    plt.subplot(n_row,n_col,index+1)
    sns.boxplot(data=df, x=value, y='OverallQual', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
condFeatures = [c for c in houses.columns if c.endswith('Cond') & (c != 'OverallCond')]

n_col = 2
n_row = int(np.ceil(len(condFeatures) / n_col))

plt.figure(figsize=(15, 10))
for index, value in enumerate(condFeatures):
    plt.subplot(n_row, n_col, index+1)
    sns.boxplot(data=houses, x=value, y='OverallCond', order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
plt.show()
plt.figure(figsize=(12, 9))
sns.heatmap(houses.corr())
plt.show()
plt.figure(figsize=(8, 7))
sns.scatterplot(houses['YearBuilt'], houses['OverallCond'])
plt.show()
houses.loc[:, ['OverallCond', 'YearBuilt']].corr()
plt.figure(figsize=(8, 7))
sns.scatterplot(houses['TotalBsmtSF'], houses['1stFlrSF'])
plt.show()
houses.loc[:, ['TotalBsmtSF', '1stFlrSF']].corr()
plt.figure(figsize=(8, 7))
sns.boxplot(data=houses, x='TotRmsAbvGrd', y='GrLivArea')
plt.show()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy import stats

class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attributes]

class MissingDummyAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        na_bools = X.isna().any()
        na_columns = na_bools[na_bools].index.values
        result = pd.DataFrame()
        for i, column in enumerate(na_columns):
            result['na_dummy_' + column] = X[column].isna()
        return result
    
class NormalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy):
        self.strategy = strategy
    
    def fit(self, X, y=None):
        if self.strategy == 'median':
            self.impute_values = X.median()
        elif self.strategy == 'mode':
            self.impute_values = X.mode().iloc[0]
        elif self.strategy == 'none':
            columns = X.columns.values
            self.impute_values = pd.Series(['None' for c in columns], index=columns)         
        elif self.strategy == 'zero':
            columns = X.columns.values
            self.impute_values = pd.Series([0 for c in columns], index=columns) 
        return self
    
    def transform(self, X, y=None):                
        for index, column in enumerate(X):  
            X[column] = X[[column]].fillna(self.impute_values[column])
        return X
    
class GroupedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, groupby):
        self.strategy = strategy
        self.groupby = groupby
    
    def fit(self, X, y=None):
        if self.strategy == 'median':
            fn = np.median
        elif self.strategy == 'mode':
            fn = (lambda x: stats.mode(x)[0][0])
        self.impute_values = X.groupby(X[self.groupby], as_index=False).agg(fn)
        return self
    
    def transform(self, X, y=None):                
        for index, column in enumerate(X):  
            if(column != self.groupby):
                X[column]  = X.apply(lambda row: self.fill(row, column), axis=1)
        return X.drop(groupby, axis=1)

    def fill(self, row, column):
        
        if(np.isnan(row[column])):
            return self.impute_values.loc[self.impute_values[self.groupby] == row[self.groupby], column].values[0]
        else:
            return row[column]

class BoolIndicesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, bools):
        self.strategy = strategy
        self.bools = bools
    
    def fit(self, X, y=None):
        if self.strategy == 'median':
            self.impute_value = X.median()
        elif self.strategy == 'mode':
            self.impute_value = X.mode().iloc[0]
        elif self.strategy == 'none':
            self.impute_value = 'none'
        elif self.strategy == 0:
            self.impute_value = 0
        return self
    
    def transform(self, X, y=None):                
        X.fillna(self.impute_value, inplace=True)
        return X     
def drop_rows(df):
    #Drop row according to  columns which has small number of missing values
    na_sums = df.isna().sum()
    columns_to_drop = na_sums[(na_sums < 10) & (na_sums != 0)].index.values
    for i, column in enumerate(columns_to_drop):
        df = df.loc[~df[column].isna(), :]
    return df
def impute_missing_values(df):
    none_impute = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']
    mode_impute = ['MasVnrType', 'MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Utilities', 'Functional', 'Exterior2nd', 'Exterior1st', 'SaleType', 'KitchenQual', 'GarageCars', 'GarageArea', 'Electrical', 'GarageYrBlt', 'TotalBsmtSF', 'BsmtUnfSF']
    median_impute = ['MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2']
    
    column_with_ns = none_impute + mode_impute + median_impute
    columnn_without_na = [c for c in df.columns.values if c not in column_with_ns]
    
    #impute missing data
    none_imputed = NormalImputer(strategy='none').fit_transform(df[none_impute])
    mode_imputed = NormalImputer(strategy='mode').fit_transform(df[mode_impute])
    median_imputed = NormalImputer(strategy='median').fit_transform(df[median_impute])
    
    return pd.concat([df[columnn_without_na], none_imputed, mode_imputed, median_imputed], axis=1)
def create_submission(preds):
    result = pd.DataFrame({
        'Id': np.arange(1461, 2919+1),
        'SalePrice' : preds
    })
    
    result.to_csv('submission/lin_with_high_correlated.csv', index=False)
X_test = pd.read_csv('../input/test.csv')

X_train = houses.drop(columns=['SalePrice'], axis=1)
y_train = houses['SalePrice']
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression

#impute missing values
X_train_prepared = impute_missing_values(X_train)
X_test_prepared = impute_missing_values(X_test)

#get categorial columns and numerical columns
cate_features = X_train_prepared.dtypes[X_train_prepared.dtypes == 'object'].index.values
num_feature = X_train_prepared.dtypes[X_train_prepared.dtypes != 'object'].drop(index='Id').index.values

#One-hot encoding for categorical columns
all_data = pd.concat([X_train_prepared, X_test_prepared], axis=0)
all_data = pd.get_dummies(data=all_data, columns=cate_features, drop_first=True)

X_train_min_Id = X_train['Id'].min()
X_train_max_Id = X_train['Id'].max()

X_test_min_Id = X_test['Id'].min()
X_test_max_Id = X_test['Id'].max()

X_train_prepared = all_data.loc[(all_data.Id >= X_train_min_Id) & (all_data.Id <= X_train_max_Id), :]
X_train_cate_dummies = X_train_prepared.loc[:, [c for c in X_train_prepared.columns if c not in num_feature]]
X_test_prepared = all_data.loc[(all_data.Id >= X_test_min_Id) & (all_data.Id <= X_test_max_Id), :]
X_test_cate_dummies = X_test_prepared.loc[:, [c for c in X_test_prepared.columns if c not in num_feature]]

#Drop Id column
X_train_prepared.drop(columns=['Id'],  inplace=True)
X_test_prepared.drop(columns=['Id'],  inplace=True)

#Rescale numerical 
X_train_num_scaled = RobustScaler().fit_transform(X_train_prepared[num_feature])
X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_feature)

X_test_num_scaled = RobustScaler().fit_transform(X_test_prepared[num_feature])
X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_feature)

#concat prepared categorical and numberical features

X_train_prepared = pd.concat([X_train_num_scaled, X_train_cate_dummies], axis=1)
X_test_prepared = pd.concat([X_test_num_scaled, X_test_cate_dummies], axis=1)


#create model
lin_reg = LinearRegression()
cv_scores = cross_val_score(estimator = lin_reg,
                                X = X_train_prepared,
                                y = y_train, 
                                cv = 10, 
                                scoring = 'neg_mean_squared_error')

print(f"CV scores: {-cv_scores.mean()}")

corrs = houses.corr()['SalePrice'].sort_values(ascending=False)
top_corrs = corrs[corrs > 0.5].drop('SalePrice')
top_corrs
top_corrs_features = list(top_corrs.index.values)
sns.heatmap(houses[top_corrs_features].corr())
features_to_discard = ['GarageArea']
features_to_train = [feature for feature in top_corrs_features if feature not in features_to_discard]

#Select categorical features that have strong correlation to SalePrice
top_corr_cate_feature = ['ExterQual', 'BsmtExposure', 'TotalBath', 'SaleType', 'SaleCondition']
features_to_train = features_to_train + top_corr_cate_feature
X_train = houses.drop(columns=['SalePrice'], axis=0)
Y_train = houses['SalePrice']

X_test = X_test.copy()

X_train['TotalBath'] = X_train['FullBath'] + X_train['BsmtFullBath'] + X_train['BsmtHalfBath'] + X_train['HalfBath']
X_test['TotalBath'] = X_test['FullBath'] + X_test['BsmtFullBath'] + X_test['BsmtHalfBath'] + X_test['HalfBath']

X_train = X_train[features_to_train]
X_test = X_test[features_to_train]


#check missing values in train set
print("Train set missing values")
X_train.isna().sum()
#impute missing values in trian set
X_train['BsmtExposure'].fillna('None', inplace=True)
#check missing values in test set
print("Test set missing values")
X_test.isna().sum()
X_test['GarageCars'].fillna(X_test['GarageCars'].median(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].median(), inplace=True)
X_test['TotalBath'].fillna(X_test['TotalBath'].median(), inplace=True)
X_test['BsmtExposure'].fillna('none', inplace=True)
X_test['SaleType'].fillna(X_test['SaleType'].mode()[0], inplace=True)

all_data = pd.concat([X_train, X_test], axis=0)
dtypes = all_data[features_to_train].dtypes
cate_features = dtypes[dtypes == 'object'].index.values
num_features = dtypes[dtypes != 'object'].index.values
all_data = pd.get_dummies(data=all_data, columns=cate_features, drop_first=True)
X_train = all_data.iloc[:X_train.shape[0]]
X_test = all_data.iloc[X_train.shape[0]: ]
X_train_num_scaled = RobustScaler().fit_transform(X_train[num_features])
X_test_num_scaled = RobustScaler().fit_transform(X_test[num_features])
X_train = np.hstack([X_train_num_scaled, X_train.loc[:, [column for column in X_train.columns if column not in num_features]]])
X_test = np.hstack([X_test_num_scaled, X_test.loc[:, [column for column in X_test.columns if column not in num_features]]])


lin = LinearRegression()
cv_scores = cross_val_score(estimator=lin,
                           X= X_train,
                           y=y_train,
                           cv=10,
                           scoring='neg_mean_squared_error' )

print(f"CV scores: {-cv_scores.mean()}")
