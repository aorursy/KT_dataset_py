# Essentials

import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000



import os

#print(os.listdir("../input/kernel-files"))
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

sample= pd.read_csv('../input/sample_submission.csv')
sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(train['SalePrice'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
# Finding numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in train.columns:

    if train[i].dtype in numeric_dtypes:

        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:

            pass

        else:

            numeric.append(i)     

# visualising some more outliers in the data values

fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))

plt.subplots_adjust(right=2)

plt.subplots_adjust(top=2)

sns.color_palette("husl", 8)

for i, feature in enumerate(list(train[numeric]), 1):

    if(feature=='MiscVal'):

        break

    plt.subplot(len(list(numeric)), 3, i)

    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)

        

    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

    plt.ylabel('SalePrice', size=15, labelpad=12.5)

    

    for j in range(2):

        plt.tick_params(axis='x', labelsize=12)

        plt.tick_params(axis='y', labelsize=12)

    

    plt.legend(loc='best', prop={'size': 10})

        

plt.show()
y = train[['Id','SalePrice']]

train_labels = train['SalePrice'].reset_index(drop=True)

train = train.drop('SalePrice',axis=1)
all_dfs = [train,test]

all_df = pd.concat(all_dfs).reset_index(drop=True);
all_df.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],axis=1,inplace=True)
all_df['LotFrontage'].fillna(value=all_df['LotFrontage'].median(),inplace=True)

all_df['MasVnrType'].fillna(value='None',inplace=True)

all_df['MasVnrArea'].fillna(0,inplace=True)

all_df['BsmtCond'].fillna(value='TA',inplace=True)

all_df['BsmtExposure'].fillna(value='No',inplace=True)

all_df['Electrical'].fillna(value='SBrkr',inplace=True)

all_df['BsmtFinType2'].fillna(value='Unf',inplace=True)

all_df['GarageType'].fillna(value='Attchd',inplace=True)

all_df['GarageYrBlt'].fillna(value=all_df['GarageYrBlt'].median(),inplace=True)

all_df['GarageFinish'].fillna(value='Unf',inplace=True)

all_df['GarageQual'].fillna(value='TA',inplace=True)

all_df['GarageCond'].fillna(value='TA',inplace=True)

all_df['BsmtFinType1'].fillna(value='NO',inplace=True)

all_df['BsmtQual'].fillna(value='No',inplace=True)

all_df['BsmtFullBath'].fillna(value=all_df['BsmtFullBath'].median(),inplace=True)

all_df['BsmtFinSF1'].fillna(value=all_df['BsmtFinSF1'].median(),inplace=True)

all_df['BsmtFinSF2'].fillna(value=0,inplace=True)

all_df['BsmtUnfSF'].fillna(value=0,inplace=True)

all_df['TotalBsmtSF'].fillna(value=all_df['TotalBsmtSF'].median(),inplace=True)

all_df['BsmtHalfBath'].fillna(value=0,inplace=True)

all_df['GarageCars'].fillna(value=all_df['GarageCars'].median(),inplace=True)

all_df['GarageArea'].fillna(value=all_df['GarageArea'].median(),inplace=True)
labelencoder=LabelEncoder()



all_df['MSZoning']      = labelencoder.fit_transform(all_df['MSZoning'].astype(str))

all_df['Exterior1st']   = labelencoder.fit_transform(all_df['Exterior1st'].astype(str))

all_df['Exterior2nd']   = labelencoder.fit_transform(all_df['Exterior2nd'].astype(str))

all_df['KitchenQual']   = labelencoder.fit_transform(all_df['KitchenQual'].astype(str))

all_df['Functional']    = labelencoder.fit_transform(all_df['Functional'].astype(str))

all_df['SaleType']      = labelencoder.fit_transform(all_df['SaleType'].astype(str))

all_df['Street']        = labelencoder.fit_transform(all_df['Street'])   

all_df['LotShape']      = labelencoder.fit_transform(all_df['LotShape'])   

all_df['LandContour']   = labelencoder.fit_transform(all_df['LandContour'])   

all_df['LotConfig']     = labelencoder.fit_transform(all_df['LotConfig'])   

all_df['LandSlope']     = labelencoder.fit_transform(all_df['LandSlope'])   

all_df['Neighborhood']  = labelencoder.fit_transform(all_df['Neighborhood'])   

all_df['Condition1']    = labelencoder.fit_transform(all_df['Condition1'])   

all_df['Condition2']    = labelencoder.fit_transform(all_df['Condition2'])   

all_df['BldgType']      = labelencoder.fit_transform(all_df['BldgType'])   

all_df['HouseStyle']    = labelencoder.fit_transform(all_df['HouseStyle'])   

all_df['RoofStyle']     = labelencoder.fit_transform(all_df['RoofStyle'])   

all_df['RoofMatl']      = labelencoder.fit_transform(all_df['RoofMatl'])    

all_df['MasVnrType']    = labelencoder.fit_transform(all_df['MasVnrType'])   

all_df['ExterQual']     = labelencoder.fit_transform(all_df['ExterQual'])  

all_df['ExterCond']     = labelencoder.fit_transform(all_df['ExterCond'])   

all_df['Foundation']    = labelencoder.fit_transform(all_df['Foundation'])   

all_df['BsmtQual']      = labelencoder.fit_transform(all_df['BsmtQual'])   

all_df['BsmtCond']      = labelencoder.fit_transform(all_df['BsmtCond'])   

all_df['BsmtExposure']  = labelencoder.fit_transform(all_df['BsmtExposure'])   

all_df['BsmtFinType1']  = labelencoder.fit_transform(all_df['BsmtFinType1'])   

all_df['BsmtFinType2']  = labelencoder.fit_transform(all_df['BsmtFinType2'])   

all_df['Heating']       = labelencoder.fit_transform(all_df['Heating'])   

all_df['HeatingQC']     = labelencoder.fit_transform(all_df['HeatingQC'])   

all_df['CentralAir']    = labelencoder.fit_transform(all_df['CentralAir'])   

all_df['Electrical']    = labelencoder.fit_transform(all_df['Electrical'])    

all_df['GarageType']    = labelencoder.fit_transform(all_df['GarageType'])  

all_df['GarageFinish']  = labelencoder.fit_transform(all_df['GarageFinish'])   

all_df['GarageQual']    = labelencoder.fit_transform(all_df['GarageQual'])  

all_df['GarageCond']    = labelencoder.fit_transform(all_df['GarageCond'])   

all_df['PavedDrive']    = labelencoder.fit_transform(all_df['PavedDrive'])  

all_df['SaleCondition'] = labelencoder.fit_transform(all_df['SaleCondition'])  
Scaler = StandardScaler()

all_scaled = pd.DataFrame(Scaler.fit_transform(all_df))



train_scaled = pd.DataFrame(all_scaled[:1460])

test_scaled = pd.DataFrame(all_scaled[1460:2920])
X = train_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y['SalePrice'], test_size=0.1, random_state=42)
kf = KFold(n_splits=12, random_state=42, shuffle=True)



# Define error metrics

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
LGBM = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# XGBoost Regressor

XGB = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)
from xgboost import XGBRegressor

#XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)

XGB.fit(X_train,y_train)
from lightgbm import LGBMRegressor

#LGBM = LGBMRegressor(n_estimators = 1000)

LGBM.fit(X_train,y_train)
print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))

print ("Training score:",LGBM.score(X_train,y_train),"Test Score:",LGBM.score(X_test,y_test))
y_pred_xgb  = pd.DataFrame( XGB.predict(test_scaled))

y_pred_lgbm = pd.DataFrame(LGBM.predict(test_scaled))



y_pred=pd.DataFrame()

y_pred['SalePrice'] = 0.5 * y_pred_xgb[0] + 0.5 * y_pred_lgbm[0]

y_pred['Id'] = test['Id']
y_pred.to_csv('house_price_blend.csv',index=False)