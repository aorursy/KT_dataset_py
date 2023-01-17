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
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import VotingRegressor



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



import re

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.info()
y     = df_train[['Id','SalePrice']]

df_train = df_train.drop('SalePrice',axis=1)
concat_df = [df_train,df_test]

df_all = pd.concat(concat_df).reset_index(drop=True)
df_all.info()
df_dummy = pd.get_dummies(df_all['MSZoning'],prefix='MSZoning',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['MSZoning']
df_all['LotFrontage'] = df_all['LotFrontage'].fillna(df_all['LotFrontage'].mean())
def LotFrontage(n):

    if (n > 21) & (n < 50):

        return 1

    elif (n > 50) & (n <= 70):

        return 2

    elif (n > 70) & (n <= 75):

        return 3

    elif (n > 75) & (n <= 80):

        return 4

    elif (n > 80) & (n <= 90):

        return 5

    elif (n > 100) & (n <= 120):

        return 6

    elif (n > 120) & (n <= 140):

        return 7

    elif (n > 140) & (n <= 160):

        return 8

    elif (n > 160) & (n <= 180):

        return 9

    elif (n > 180) & (n <= 200):

        return 9

    else:

        return 10
df_all['LotFrontage'] = df_all['LotFrontage'].apply(LotFrontage)
df_dummy = pd.get_dummies(df_all['Neighborhood'],prefix='Neighborhood',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Neighborhood']
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna('Other')

df_dummy = pd.get_dummies(df_all['Exterior2nd'],prefix='Exterior2nd',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Exterior2nd']
df_dummy = pd.get_dummies(df_all['Utilities'],prefix='Utilities',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Utilities']
df_dummy = pd.get_dummies(df_all['LandContour'],prefix='LandContour',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['LandContour']
df_dummy = pd.get_dummies(df_all['Condition1'],prefix='Condition1',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Condition1']
def Condition2(c):

    if c == 'Norm':

        return 1

    elif c == 'Feedr':

        return 2

    elif c == 'PosA':

        return 3

    elif c == 'PosN':

        return 4

    elif c == 'Artery':

        return 5

    else:

        return 0
df_all['Condition2'] = df_all['Condition2'].apply(Condition2).astype(int)
df_dummy = pd.get_dummies(df_all['BldgType'],prefix='BldgType',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['BldgType']
df_dummy = pd.get_dummies(df_all['HouseStyle'],prefix='HouseStyle',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['HouseStyle']
df_dummy = pd.get_dummies(df_all['RoofMatl'],prefix='RoofMatl',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['RoofMatl']
df_all['Exterior1st']= df_all['Exterior1st'].fillna('Other')
def Exterior1st(c):

    if c == 'VinylSd':

        return 1

    elif c == 'Wd Sdng':

        return 2

    elif c == 'HdBoard':

        return 3

    elif c == 'Plywood':

        return 4

    elif c == 'MetalSd':

        return 5

    elif c == 'CemntBd':

        return 6

    elif c == 'WdShing':

        return 7

    elif c == 'BrkFace':

        return 8

    elif c == 'AsbShng':

        return 9

    else:

        return 0
df_all['Exterior1st'] = df_all['Exterior1st'].apply(Exterior1st).astype(int)
def BsmtFinSF1(n):

    if (n > -1) & (n <= 150):

        return 1

    elif (n > 150) & (n <= 300):

        return 2

    elif (n > 300) & (n <= 450):

        return 3

    elif (n > 450) & (n <= 600):

        return 4

    elif (n > 600) & (n <= 750):

        return 5

    else:

        return 10
df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].apply(BsmtFinSF1).astype(int)
def MasVnrArea(n):

    if (n > -1) & (n <= 150):

        return 1

    elif (n > 150) & (n <= 200):

        return 2

    elif (n > 200) & (n <= 300):

        return 3

    else:

        return 0
df_all['MasVnrArea'] = df_all['MasVnrArea'].apply(MasVnrArea).astype(int)
df_dummy = pd.get_dummies(df_all['ExterQual'],prefix='ExterQual',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['ExterQual']
df_dummy = pd.get_dummies(df_all['ExterCond'],prefix='ExterCond',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['ExterCond']
df_all['BsmtQual'] = df_all['BsmtQual'].fillna('TA')

df_dummy = pd.get_dummies(df_all['BsmtQual'],prefix='BsmtQual',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['BsmtQual']
df_all['BsmtCond'] = df_all['BsmtCond'].fillna('TA')

df_dummy = pd.get_dummies(df_all['BsmtCond'],prefix='BsmtCond',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['BsmtCond']
df_all['BsmtExposure'] = df_all['BsmtExposure'].fillna('TA')

df_dummy = pd.get_dummies(df_all['BsmtExposure'],prefix='BsmtExposure',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['BsmtExposure']
def Functional(f):

    if f == 'Typ':

        return 1

    elif f == 'Min1':

        return 2

    elif f == 'Min2':

        return 3

    elif f == 'Mod':

        return 4

    elif f == 'Maj1':

        return 5

    elif f == 'Maj2':

        return 6

    elif f == 'Sev':

        return 7

    elif f == 'Sal':

        return 8

    else:

        return 0
df_all['Functional'] = df_all['Functional'].apply(Functional).astype(int)
df_dummy = pd.get_dummies(df_all['GarageType'],prefix='GarageType',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['GarageType']
df_dummy = pd.get_dummies(df_all['SaleType'],prefix='SaleType',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['SaleType']
df_dummy = pd.get_dummies(df_all['SaleCondition'],prefix='SaleCondition',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['SaleCondition']
df_dummy = pd.get_dummies(df_all['Heating'],prefix='Heating',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Heating']
df_dummy = pd.get_dummies(df_all['KitchenQual'],prefix='KitchenQual',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['KitchenQual']
df_all['GarageQual'] = df_all['GarageQual'].fillna('TA')

df_dummy = pd.get_dummies(df_all['GarageQual'],prefix='GarageQual',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['GarageQual']
df_dummy = pd.get_dummies(df_all['Foundation'],prefix='Foundation',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['Foundation']
df_dummy = pd.get_dummies(df_all['GarageCond'],prefix='GarageCond',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['GarageCond']
df_dummy = pd.get_dummies(df_all['PavedDrive'],prefix='PavedDrive',dummy_na=True)

df_all = pd.concat([df_all,df_dummy],axis=1)

del df_all['PavedDrive']
df_all['GarageYrBlt']= df_all['GarageYrBlt'].fillna('2003').astype(int)

df_all['TotalBsmtSF'] = df_all['TotalBsmtSF'].fillna('1305').astype(int)

df_all['BsmtFullBath'] = df_all['BsmtFullBath'].fillna('0').astype(int)

df_all['BsmtHalfBath'] = df_all['BsmtHalfBath'].fillna('0').astype(int)

df_all['GarageCars'] = df_all['GarageCars'].fillna('2').astype(int)

df_all['GarageArea'] = df_all['GarageArea'].fillna('420').astype(int)

df_all['PoolArea']=df_all['PoolArea'].astype(int)
df_all =df_all.drop(['YrSold','MoSold','MiscVal','PoolQC','ScreenPorch','3SsnPorch','EnclosedPorch',

                     'KitchenAbvGr','CentralAir','Alley','LotConfig','LotShape','LandSlope','RoofStyle',

                     'BsmtFinSF2','BsmtUnfSF','HeatingQC','Electrical','BsmtFinType2','BsmtFinType1',

                     'GarageFinish','MiscFeature','FireplaceQu','Fence','Alley','Street','MasVnrType'], axis = 1)
Scaler = StandardScaler()

all_scaled = pd.DataFrame(Scaler.fit_transform(df_all))



train_scaled = pd.DataFrame(all_scaled[:1460])

test_scaled = pd.DataFrame(all_scaled[1460:2920])
X_train = train_scaled

y_train = y['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(

    X_train, y_train, test_size=0.15, random_state=0)
XGBoost

param_xgb = {"max_depth": [1,3,5,10,100],

             "learning_rate" : [0.0001,0.001,0.01],

             "min_child_weight" : [1, 3, 5, 10],

             "n_estimators": [1, 10, 100, 1000],

             "subsample": [0.5,0.75,0.9],

             "gamma":[0,0.1,0.2],

             "eta": [0.3,0.15,0.10]}



gs_xgb = GridSearchCV(XGBRegressor(),

                      param_xgb,

                      cv=4,#cross validation

                      verbose=True,#Display Logs

                      n_jobs=-1)#Multi Tasking

gs_xgb.fit(X_train, y_train)

 

print(gs_xgb.best_estimator_)
LightGBM

param_lgb = {"max_depth": [1, 3, 5, 10, 25, 50, 75],

             "learning_rate" : [0.001,0.01,0.05,0.1],

             "num_leaves": [1, 10, 100,1000],

             "n_estimators": [1, 3, 10, 100, 1000]}



gs_lgb = GridSearchCV(LGBMRegressor(),

                      param_lgb,

                      cv=4,#cross validation

                      verbose=True,#Display Logs

                      n_jobs=-1)#Multi Tasking

gs_lgb.fit(X_train, y_train)



print(gs_lgb.best_estimator_)
XGB = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                   colsample_bynode=1, colsample_bytree=1, eta=0.3, gamma=0,

                   gpu_id=-1, importance_type='gain', interaction_constraints='',

                   learning_rate=0.01, max_delta_step=0, max_depth=5,

                   min_child_weight=1,monotone_constraints='()',n_estimators=1000,

                   n_jobs=0, num_parallel_tree=1, random_state=0,

                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.5,

                   tree_method='exact', validate_parameters=1, verbosity=None)

XGB.fit(X_train,y_train)
model_lgb = LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=1000, num_leaves=10)

model_lgb.fit(X_train , y_train)
votingC = VotingRegressor(estimators=[('xgb_r', XGB),('lgb_r', model_lgb)],n_jobs=4)



votingC = votingC.fit(X_train, y_train)
print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))

print ("Training score:",model_lgb.score(X_train,y_train),"Test Score:",model_lgb.score(X_test,y_test))

print ("Training score:",votingC.score(X_train,y_train),"Test Score:",votingC.score(X_test,y_test))
y_pred_voting = pd.DataFrame(votingC.predict(test_scaled))
y_pred=pd.DataFrame()

y_pred['SalePrice'] = y_pred_voting[0]

y_pred['Id'] = df_test['Id']
y_pred.to_csv('submission.csv',index=False)