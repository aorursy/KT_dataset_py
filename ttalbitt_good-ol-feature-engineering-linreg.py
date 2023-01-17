# pre-processing functions
def exterQual(e):
    if e == 'Ex':
        return 3.5
    elif e == 'Gd':
        return 2
    elif e == 'TA':
        return 1
    elif e == 'Fa':
        return 0
    else:
        return -.5

def garage_bad(g):
    if g['GarageFinish'] == 'Unf' or g['GarageCond'] == 'Fa' or g['GarageQual'] == 'Fa' or g['GarageCond'] == 'Po' or g['GarageQual'] == 'Po':
        return 1
    else:
        return 0
    
    
def feature_engineering(X):
    columns_to_drop = ['Street','Alley','Condition2','RoofStyle','RoofMatl','Exterior2nd','Heating','SaleType','FireplaceQu','Functional','KitchenQual','Electrical','Utilities',
                      'LotFrontage', 'BsmtFinSF2', 'TotalBsmtSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'MoSold']

    X['ExterQual'] = X['ExterQual'].apply(exterQual)
    X['ExterCond'] = X['ExterCond'].apply(lambda x : int(x == 'Po' or x == 'Fa'))
    X['Exterior_bad'] = X['Exterior1st'].apply(lambda x : int(x == 'AsbShng' or x== 'CBlock'))
    X['Exterior_stucco'] = X['Exterior1st'].apply(lambda x : int(x == 'Stucco'))
    X['Exterior_wd'] = X['Exterior1st'].apply(lambda x : int(x == 'HdBoard' or x == 'Plywood' or x == 'Wd Sdng'))
    X['Exterior_hard'] = X['Exterior1st'].apply(lambda x : int(x == 'MetalSd' or x == 'CemntBd'))
    X['Exterior_vinyl'] = X['Exterior1st'].apply(lambda x : int(x == 'VinylSd'))
    columns_to_drop.append('Exterior1st')
    
    X['BsmtQual'] = X['BsmtQual'].apply(lambda x : int(x == 'Ex'))
    X['BsmtCond'] = X['BsmtCond'].apply(lambda x : int(x == 'Po' or x == 'Fa'))
    X['BsmtExposure'] = X['BsmtExposure'].apply(lambda x : int(x == 'Gd' or x == 'Av'))
    X['BsmtFinType1'] = X['BsmtFinType1'].apply(lambda x : int(x == 'GLQ'))
    columns_to_drop.append('BsmtFinType2')
    
    X['HeatingQC_bad'] = X['HeatingQC'].apply(lambda x : int(x == 'Fa' or x == 'Po'))
    X['HeatingQC_ex'] = X['HeatingQC'].apply(lambda x : int(x == 'Ex'))
    columns_to_drop.append('HeatingQC')
    
    X['Pool'] = X['PoolQC'].apply(lambda x : int(pd.isna(x)))
    columns_to_drop.append('PoolQC')
    columns_to_drop.append('PoolArea')

    X['PavedDrive'] = X['PavedDrive'].apply(lambda x : int(x == 'Y'))

    X['Garage_bad'] = X[['GarageQual', 'GarageCond', 'GarageFinish']].apply(garage_bad, axis=1)
    X['GarageType'] = X['GarageType'].apply(lambda x : int(x == 'Detchd' or x == 'CarPort' or pd.isna(x)))
    X.drop(['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'], axis=1, inplace=True)

    X['Fireplaces'] = X['Fireplaces'].apply(lambda x : int(x >= 1))
    
    X['MiscFeature'] = X['MiscFeature'].apply(lambda x : int(pd.isna(x)))
    X['SaleCondition'] = X['SaleCondition'].apply(lambda x : int(x == 'Partial'))
    X['Fence'] = X['Fence'].apply(lambda x : int(x == 'MnWw' or x == 'MnPrv'))
    X['MasVnrType'] = X['MasVnrType'].apply(lambda x : int(x == 'Stone' or x == 'BrkFace'))
    X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
    X['Foundation'] = X['Foundation'].apply(lambda x : int(x == 'PConc'))
    X['LandSlope'] = X['LandSlope'].apply(lambda x : int(x == 'Gtl'))
    X['CentralAir'] = X['CentralAir'].apply(lambda x : int(x == 'Y'))
    
    X['HouseStyle'] = X['HouseStyle'].apply(lambda x : int(x == '1Story' or x == '2Story'))

    X['MSZ_rh'] = X['MSZoning'].apply(lambda x : int(x == 'RH'))
    X['MSZ_rm'] = X['MSZoning'].apply(lambda x : int(x == 'RM'))
    X['MSZ_rl'] = X['MSZoning'].apply(lambda x : int(x == 'RL'))
    columns_to_drop.append('MSZoning')
    
    X['LotShape'] = X['LotShape'].apply(lambda x : int(x == 'Reg'))
    X['LotConfig'] = X['LotConfig'].apply(lambda x : int(x == 'Corner' or x == 'CulDSaq'))
    X['LandContour'] = X['LandContour'].apply(lambda x : int(x == 'Lvl'))
    columns_to_drop.append('LandContour')
    
    X['Con1_art'] = X['Condition1'].apply(lambda x : int(x == 'Artery'))
    X['Con1_pos'] = X['Condition1'].apply(lambda x : int(x == 'PosN' or x == 'PosA'))
    X['Con1_rr'] = X['Condition1'].apply(lambda x : int('RR' in x))
    columns_to_drop.append('Condition1')
    
    X['Twnhs'] = X['BldgType'].apply(lambda x : int('Twnhs' in x))    
    columns_to_drop.append('BldgType')

    X = pd.get_dummies(X, columns=['Neighborhood'])
    
    X['Overall'] = X['OverallQual'] + X['OverallCond']
    columns_to_drop.append('OverallQual')
    columns_to_drop.append('OverallCond')
    
    X['YrSold'] = X['YrSold'].apply(lambda x : int(x < 2008))
    
    X['EncPorch'] = X[['ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'OpenPorchSF']].apply(lambda x : x['ScreenPorch'] + x['3SsnPorch'] + x['EnclosedPorch'] + x['OpenPorchSF'], axis=1)
    columns_to_drop.append('ScreenPorch')
    columns_to_drop.append('3SsnPorch')
    columns_to_drop.append('EnclosedPorch')
    columns_to_drop.append('OpenPorchSF')
    
    X = X.drop(columns_to_drop, axis=1)
    
    return X
# Data Pre-Processing
import pandas as pd
pd.options.mode.chained_assignment = None

#import numpy as np

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_train = df.drop('SalePrice', axis=1)
y_train = df['SalePrice']

X_train = feature_engineering(X_train)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_test = feature_engineering(X_test)
X_test = X_test.fillna(0)
y_pred = rfr.predict(X_test)
sol = pd.DataFrame(y_pred, columns=['SalePrice'])
sol.index += 1461
sol['Id'] = sol.index
sol.index = sol['Id']
sol.drop('Id', axis=1, inplace=True)

sol.to_csv('submission.csv')