# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test['Id']
#train.info()
print (train.shape[0])
print(test.shape[0])
#train.head(5)
train_feature = train.drop('SalePrice',axis=1)
train_label = train['SalePrice'].copy()
data = pd.concat([train_feature,test],sort=False)
data.describe()
data.info()
data.drop('Id',axis=1,inplace=True)
# some features have almost all nan in their fields,so drop the features
lose_feature = ['Alley','PoolQC','MiscFeature','Fence']
train_del_nan_feature = data.drop(lose_feature,axis=1)
obj_features = list(train_del_nan_feature.select_dtypes(include='object').columns.values)
num_features = list(train_del_nan_feature.select_dtypes(exclude='object').columns.values)
print(len(obj_features))
print(len(num_features))
#num_feature = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
#              'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch'
#              '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']
#obj_feature = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
#              'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',
#              'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',]
#train
print(train_del_nan_feature.shape)
num_data = train_del_nan_feature[num_features]
obj_data = train_del_nan_feature[obj_features]
#split the feature of numerical data and categrical data
'''print (train_set_X.shape)
print (train_set_y.shape)
print (test_set_X.shape)
print (test_set_y.shape)
print (validation_set_X.shape)
print (validation_set_y.shape)
'''
print(num_data.shape)
print(obj_data.shape)
corr_matrix = train.corr()
corr_series_with_label = corr_matrix['SalePrice'].sort_values(ascending=False)
f, ax = plt.subplots(figsize=(12, 9))
import seaborn as sns
sns.heatmap(corr_matrix, vmax=.8, square=True);
corr_threshold = 0.5
print(corr_series_with_label.head(11))
name_list = []
for name,_ in corr_series_with_label.head(11).items():
    name_list.append(name)
print (name_list)
%matplotlib inline
from pandas.plotting import scatter_matrix
scatter_matrix(train[name_list],figsize=(20,12))
plt.show()
num_data['OverallQual'].value_counts() #Over the 10 features with highest corr with the labels ,the OverallQual and GarageCars are Discrete variables
#[GrLivArea,GarageArea,TotalBsmtSF,1stFlrSF] but you can see that from the Year built on ,the corr is really weak.
train.plot(kind='scatter',x='PoolArea',y='SalePrice',alpha=0.1)
num_data['PoolArea'].value_counts()
# the first thing I need to do is to eliminate the feature which is meaningless
# this step is very tricky and emm...
#for index in corr.indx:
#    for co
threshold = 0.8
_ = corr_matrix[corr_matrix.apply(lambda x: x>threshold,axis=1)]
# there are some very high corr pairs [(GarageYrBlt,YearBuilt),(1stFlrSF,TotalBsmtSF),(TotRmsAbvGrd,GrLivArea)]
# delete one of them in pairs.
high_corr_feature_list = ['GarageYrBlt','1stFlrSF','TotRmsAbvGrd']
num_data_without_high_corr = num_data.drop(high_corr_feature_list,axis=1)
print (num_data_without_high_corr.shape)
num_data_without_high_corr.columns
#num_data_without_high_corr.info()
#num_train = num_data_without_high_corr.drop('SalePrice',axis=1)
#num_label = num_data_without_high_corr['SalePrice'].copy()
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='median')
imp.fit(num_data_without_high_corr)
num_train_clean = imp.fit_transform(num_data_without_high_corr)
pd.DataFrame(num_train_clean).info()
obj_data.describe()
obj_data['Neighborhood'].value_counts()
obj_data.columns
for c in list(obj_data.columns):
    print (c + "->" + '(' + str(obj_data[c].unique()) + ')')
obj_data['MSZoning'].value_counts().max()
# we can see that many catogrical features have a phenomenon that one level can get a percentile more that 90%
#so we deveide the catogorical by (A, other)
obj_data_dummies = pd.get_dummies(obj_data)
obj_data_dummies_clean = obj_data_dummies.fillna(obj_data_dummies.mean()) # nothing fancy
obj_data_dummies_clean.values.shape
#full_data_clean = pd.concat([pd.DataFrame(num_train_clean),obj_data_dummies_clean],axis=1)
#print (full_data_clean.values.shape)
print(num_train_clean.shape)
print(obj_data_dummies_clean.values.shape)
full_data = np.concatenate([num_train_clean,obj_data_dummies_clean.values],axis=1)
full_data.shape
full_data_pd = pd.DataFrame(full_data)
full_data_pd.head(10)
train_data_clean = full_data[:train.shape[0]]
test_data_clean = full_data[train.shape[0]:]
print(train_data_clean.shape)
print(test_data_clean.shape)
# preprocess the label
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
train_label = np.log1p(train_label)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_label)
    rmse= np.sqrt(-cross_val_score(model, train_data_clean, train_label, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb.fit(train_data_clean, train_label)
lgb_pred = np.expm1(model_lgb.predict(test_data_clean))
sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = lgb_pred
sub.to_csv('submission.csv',index=False)