from xgboost.sklearn import XGBRegressor

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler
# 导入数据

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train
# 处理缺失值1.0

train.apply(lambda x: sum(x.isnull()))[:30]
test.dtypes[:30]
test['MSZoning']= test['MSZoning'].replace(np.nan,value='others')

test['Exterior1st']= test['Exterior1st'].replace(np.nan,value='others')

test['Exterior2nd']= test['Exterior2nd'].replace(np.nan,value='others')
test['Utilities']= test['Utilities'].replace(np.nan,value='others')
train['LotFrontage'].fillna(train['LotFrontage'].median(),inplace=True)

test['LotFrontage'].fillna(test['LotFrontage'].median(),inplace=True)
train['Alley'].unique()
train['Alley'] = train['Alley'].apply(lambda x: 0 if x not in ['Grvl','Pave'] else 1) # 把Alley列转换成0，1表示

test['Alley'] = test['Alley'].apply(lambda x: 0 if x not in ['Grvl','Pave'] else 1) # 把Alley列转换成0，1表示
train = train.rename(columns={'Alley':'AlleyExist'})   # 改个列名

test = test.rename(columns={'Alley':'AlleyExist'})   # 改个列名
train['MasVnrType'].unique()
# train['MasVnrType'] = train['MasVnrType'].replace('None',value = np.nan)  # 把MasVnrType列中的‘None’改成nan
# train['MasVnrType'].unique()
train['MasVnrType']= train['MasVnrType'].replace(np.nan,value='others')

test['MasVnrType']= test['MasVnrType'].replace(np.nan,value='others')
train['MasVnrType'].unique()
train['MasVnrArea']= train['MasVnrArea'].replace('others',value = np.nan)

test['MasVnrArea']= test['MasVnrArea'].replace('others',value = np.nan)
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(),inplace=True)

test['MasVnrArea'].fillna(test['MasVnrArea'].mean(),inplace=True)

# 处理缺失值2.0

test.apply(lambda x: sum(x.isnull()))[29:60]
test.dtypes[29:60]
train['Electrical'].unique()
train['Electrical'] = train['Electrical'].replace(np.nan,'others')
np.any(pd.isnull(train['Electrical']))
train['BsmtQual'] = train['BsmtQual'].replace(np.nan,'others')

test['BsmtQual'] = test['BsmtQual'].replace(np.nan,'others')
test['BsmtQual'].unique()
train['BsmtCond'] = train['BsmtCond'].replace(np.nan,'others')

test['BsmtCond'] = test['BsmtCond'].replace(np.nan,'others')
test['BsmtCond'].unique()
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(),inplace=True)

test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(),inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(),inplace=True)

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(),inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean(),inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean(),inplace=True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(),inplace=True)
test['KitchenQual'] = test['KitchenQual'].replace(np.nan,'others')

test['Functional'] = test['Functional'].replace(np.nan,'others')
test['FireplaceQu'] = test['FireplaceQu'].replace(np.nan,'others')

test['GarageType'] = test['GarageType'].replace(np.nan,'others')
train['BsmtExposure'] = train['BsmtExposure'].replace(np.nan,'others')

test['BsmtExposure'] = test['BsmtExposure'].replace(np.nan,'others')
train['BsmtFinType1'] = train['BsmtFinType1'].replace(np.nan,'others')

test['BsmtFinType1'] = test['BsmtFinType1'].replace(np.nan,'others')
train['BsmtFinType2'] = train['BsmtFinType2'].replace(np.nan,'others')

test['BsmtFinType2'] = test['BsmtFinType2'].replace(np.nan,'others')
train['FireplaceQu'] = train['FireplaceQu'].replace(np.nan,'others')
train['GarageType'] = train['GarageType'].replace(np.nan,'others')
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(),inplace=True)   # to do!,,,,,,,,,,,,,,,,,,,,
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(),inplace=True)
np.all(pd.notnull(train['GarageYrBlt']))
# 处理缺失值3.0

train.apply(lambda x: sum(x.isnull()))[55:]
test.dtypes[59:]
train['GarageFinish'] = train['GarageFinish'].replace(np.nan,'others')

test['GarageFinish'] = test['GarageFinish'].replace(np.nan,'others')
train['GarageQual'] = train['GarageQual'].replace(np.nan,'others')

test['GarageQual'] = test['GarageQual'].replace(np.nan,'others')
train['GarageCond'] = train['GarageCond'].replace(np.nan,'others')

test['GarageCond'] = test['GarageCond'].replace(np.nan,'others')
test['PoolQC'].unique()
train['PoolQC'] = train['PoolQC'].apply(lambda x: 0 if x not in [ 'Ex', 'Fa', 'Gd'] else 1 )

test['PoolQC'] = test['PoolQC'].apply(lambda x: 0 if x not in [ 'Ex', 'Fa', 'Gd'] else 1 )
train = train.rename(columns={'PoolQC':'PoolQCExist'})   # 改个列名

test = test.rename(columns={'PoolQC':'PoolQCExist'})   # 改个列名
test['Fence'].unique()
train['Fence']=train['Fence'].apply(lambda x: 0 if x not in ['MnPrv', 'GdWo', 'GdPrv', 'MnWw'] else 1 )

test['Fence']=test['Fence'].apply(lambda x: 0 if x not in ['MnPrv', 'GdWo', 'GdPrv', 'MnWw'] else 1 )
train = train.rename(columns={'Fence':'FenceExist'})

test = test.rename(columns={'Fence':'FenceExist'})
test['MiscFeature'].unique()
train['MiscFeature']= train['MiscFeature'].apply(lambda x: 1 if x in [ 'Shed', 'Gar2', 'Othr', 'TenC'] else 0)

test['MiscFeature']= test['MiscFeature'].apply(lambda x: 1 if x in [ 'Shed', 'Gar2', 'Othr', 'TenC'] else 0)
train = train.rename(columns ={'MiscFeature':'MiscFeatureExist'})

test = test.rename(columns ={'MiscFeature':'MiscFeatureExist'})
test['GarageCars'].fillna(test['GarageCars'].mean(),inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mean(),inplace=True)
test['SaleType'] = test['SaleType'].replace(np.nan,'others')
train.apply(lambda x: sum(pd.isnull(x)))[50:]
from sklearn.preprocessing import LabelEncoder  # 做特征预处理
numeric_feats = train.dtypes[train.dtypes != "object"].index
numeric_feats
le = LabelEncoder()

var_to_encode = [ 'MSZoning', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating',

       'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',

       'Functional','FireplaceQu', 'GarageType','GarageFinish','GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
for col in var_to_encode:

    train[col] = le.fit_transform(train[col])

    test[col] = le.fit_transform(test[col])
# 数据分割

x_train = train.iloc[:,1:-1]

y_train = train['SalePrice']

x_test = test.iloc[:,1:]
# 特征工程

# 实例化一个转换器

transfer= StandardScaler()

# 数据标准化

x_train = transfer.fit_transform(x_train)

x_test = transfer.fit_transform(x_test)
xgb = XGBRegressor(eval_matric='rmse')

xgb.fit(x_train,y_train)
pre = xgb.predict(x_test)
pre
from sklearn.model_selection import GridSearchCV
xgbreg = XGBRegressor({'n_estimators': 5000,

                            'subsample': 1,

                            'min_child_weight':3,

                            'gamma':0.25,

                            'objective': 'reg:squarederror','eval_matric':'rmse'})

parameters = {'learning_rate': [0.01, 0.02],

             'max_depth': [i for i in range(10,11)],'seed':[20,19,18]}

xgb_reg = GridSearchCV(estimator=xgbreg, param_grid=parameters, cv=5, n_jobs=-1).fit(x_train, y_train)
print("Best parameters set:", xgb_reg.best_params_)
xgb_reg_y_pred = xgb_reg.predict(x_test)

xgb_reg_y_pred
pre1=pd.DataFrame(xgb_reg_y_pred)

# pre1.to_csv('F:/Download/HomePrize_XGB.csv',header=0)