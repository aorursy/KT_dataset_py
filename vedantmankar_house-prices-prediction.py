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
import matplotlib.pyplot as plt
import seaborn  as sns
%matplotlib inline
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.head()
train_df.describe()
train_df.shape
train_df.info()
train_df.isna().sum()
sns.heatmap(train_df.isnull(),cbar=False)
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
train_df = train_df.drop(['Alley'],axis = 1)
train_df['MasVnrType'].value_counts()
train_df['MasVnrType'] = train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean())
train_df['BsmtQual'] = train_df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])
train_df['BsmtCond'] = train_df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])


train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna(train_df['BsmtFinType1'].mode()[0])
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])
train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])
train_df['GarageType'] = train_df['GarageType'].fillna(train_df['GarageType'].mode()[0])
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].mean())
train_df['GarageFinish'] = train_df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])
train_df = train_df.drop(['PoolQC','Fence','MiscFeature','GarageYrBlt'],axis = 1)
train_df.drop(['Id'],axis = 1,inplace=True)
train_df['GarageQual'] = train_df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])
train_df['GarageCond'] = train_df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])
train_df.shape
sns.heatmap(train_df.isnull(),cbar=False)
#categorical columns
columns = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
print(len(columns))

test_df.head()
test_df.shape
test_df.isna().sum()
test_df.info()
sns.heatmap(test_df.isnull(),cbar=False)
#filling the missing values in the test dataset
test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

test_df.shape
test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])

test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])

test_df = test_df.drop(['Alley','PoolQC'],axis=1)
test_df.shape
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mean())


test_df = test_df.drop(['Fence','MiscFeature','Id','GarageYrBlt'],axis =1)
test_df.shape
sns.heatmap(test_df.isnull(),cmap='viridis')
test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])
test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
test_df.loc[:,test_df.isnull().any()].head()
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df.shape

test_df.head()
final_df = pd.concat([train_df,test_df],axis = 0)
final_df.shape
def one_hot_multicolumn(multi_col):
    df_final = final_df
    i =0
    for col in multi_col:
        print(col)
        df = pd.get_dummies(final_df[col],drop_first=True)
        final_df.drop([col],axis = 1,inplace=True)
        if i == 0:
            df_final= df.copy()
        else:
            df_final = pd.concat([df_final,df],axis = 1)
        i = i + 1
    df_final = pd.concat([final_df,df_final],axis = 1)
    return df_final
    
#taking a copy of training data
train_copy_df = train_df.copy()
final_df = one_hot_multicolumn(columns)
final_df.shape

#removing the duplicates
final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df
df_training = final_df.iloc[:1460,:]
df_testing = final_df.iloc[1460:,:]
df_training.head()
df_testing.head()
df_testing.drop(['SalePrice'],axis = 1,inplace=True)

x_train = df_training.drop(['SalePrice'],axis = 1)
y_train = df_training['SalePrice']
regressor = xgb.XGBRegressor()
booster = ['gbtree','gblinear']
base_score = [0.20,0.25,0.50,0.75,1]
n_estimators = [100,150,200,500,700,900]
max_depth = [2,5,7,10,15]
learning_rate = [0.005,0.01,0.15,0.20]
hyperparameter_grid = {
    'n_estimators':n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'booster':booster,
    'base_score':base_score
}
random_cv = RandomizedSearchCV(estimator=regressor,
                              param_distributions=hyperparameter_grid,
                              cv = 5,
                              n_iter = 50,
                              random_state=42,
                              return_train_score=True,n_jobs = -1,
                               scoring='neg_mean_absolute_error'
                              )
random_cv.fit(x_train,y_train)
random_cv.best_estimator_
regressor = xgb.XGBRegressor(base_score=0.2, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.15, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=700, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(x_train,y_train)
import pickle
filename = 'best_xgbmodel.pkl'
pickle.dump(regressor,open(filename,'wb'))
y_pred = regressor.predict(df_testing)
y_pred
#sample submission file 
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
dataset = pd.concat([sub_df['Id'],pred],axis = 1)
dataset.columns=['Id','SalePrice']
dataset.to_csv("sample_submission.csv",index=False)
