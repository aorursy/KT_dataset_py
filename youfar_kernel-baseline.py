import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# full_df=pd.concat([train_df,test_df], ignore_index=True)
# #outlier

# corrmat = train_df.corr()

# f, ax = plt.subplots(figsize=(20, 9))

# sns.heatmap(corrmat, vmax=0.8, square=True, cmap=sns.cm.rocket_r)
# fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(40, 9))

# axes = np.ravel(axes)

# cols = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageCars', 'YearBuilt', 'GarageArea', 'FullBath',

#        'YearRemodAdd', 'TotRmsAbvGrd']

# for i, c in zip(range(10), cols):

#     train_df.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')
train_df = train_df[train_df['TotalBsmtSF'] < 3000]

train_df = train_df[train_df['1stFlrSF'] < 2500]

train_df = train_df[train_df['GrLivArea'] < 4000]
full_df=pd.concat([train_df,test_df], ignore_index=True)

#missing value

missing = full_df.isnull().sum()

missing[missing>0].sort_values(ascending=False)
cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]

full_df = full_df.drop(cols1, axis=1)


col=['GarageType','GarageQual','GarageCond','GarageFinish','BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType']

full_df[col]=full_df[col].fillna('None')





cols=['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','MasVnrArea']

full_df[cols]=full_df[cols].fillna(0,axis=1)

full_df
full_df.loc[full_df['LotFrontage'].isnull(),'LotFrontage']=full_df['LotFrontage'].mean(skipna=True)

full_df.loc[full_df['GarageYrBlt'].isnull(),'GarageYrBlt']=full_df[full_df['GarageYrBlt'].isnull()]['YearBuilt']
#!!!

cols1 = ["MSZoning", "Utilities", "Functional", "SaleType", "KitchenQual", "Exterior2nd", "Exterior1st", "Electrical"]

full_df = full_df.drop(cols1, axis=1)
missing = full_df.isnull().sum()

missing[missing>0].sort_values(ascending=False)
# from sklearn.impute import SimpleImputer

# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #for median imputation replace 'mean' with 'median'

# imp_mean.fit(full_df)

# imputed_full_df = imp_mean.transform(full_df)

# imputed_full_df
#feature_engineering

BsmtQualDf = pd.get_dummies( full_df['BsmtQual'] , prefix='BsmtQual' )

full_df = pd.concat([full_df,BsmtQualDf],axis=1)

full_df.drop('BsmtQual',axis=1,inplace=True)



BldgTypeDf = pd.get_dummies( full_df['BldgType'] , prefix='BldgType' )

full_df = pd.concat([full_df,BldgTypeDf],axis=1)

full_df.drop('BldgType',axis=1,inplace=True)



BsmtCondDf = pd.get_dummies( full_df['BsmtCond'] , prefix='BsmtCond' )

full_df = pd.concat([full_df,BsmtCondDf],axis=1)

full_df.drop('BsmtCond',axis=1,inplace=True)



BsmtExposureDf = pd.get_dummies( full_df['BsmtExposure'] , prefix='BsmtExposure' )

full_df = pd.concat([full_df,BsmtExposureDf],axis=1)

full_df.drop('BsmtExposure',axis=1,inplace=True)



BsmtFinType1Df = pd.get_dummies( full_df['BsmtFinType1'] , prefix='BsmtFinType1' )

full_df = pd.concat([full_df,BsmtFinType1Df],axis=1)

full_df.drop('BsmtFinType1',axis=1,inplace=True)



BsmtFinType2Df = pd.get_dummies( full_df['BsmtFinType2'] , prefix='BsmtFinType2' )

full_df = pd.concat([full_df,BsmtFinType2Df],axis=1)

full_df.drop('BsmtFinType2',axis=1,inplace=True)



##

# BsmtQual         object

# CentralAir       object

# Condition1       object

# Condition2       object

# ExterCond        object

# ExterQual        object

# Foundation       object

# GarageCond       object

# GarageFinish     object

# GarageQual       object

# GarageType       object

# Heating          object

# HeatingQC        object

# HouseStyle       object

# LandContour      object

# LandSlope        object

# LotConfig        object

# LotShape         object

# MasVnrType       object

# Neighborhood     object

# PavedDrive       object

# RoofMatl         object

# RoofStyle        object

# SaleCondition    object

# Street           object



CentralAirDf = pd.get_dummies( full_df['CentralAir'] , prefix='BsmtQual' )

full_df = pd.concat([full_df,CentralAirDf],axis=1)

full_df.drop('CentralAir',axis=1,inplace=True)



Condition1Df = pd.get_dummies( full_df['Condition1'] , prefix='Condition1' )

full_df = pd.concat([full_df,Condition1Df],axis=1)

full_df.drop('Condition1',axis=1,inplace=True)



Condition2Df = pd.get_dummies( full_df['Condition2'] , prefix='Condition2' )

full_df = pd.concat([full_df,Condition2Df],axis=1)

full_df.drop('Condition2',axis=1,inplace=True)



ExterCondDf = pd.get_dummies( full_df['ExterCond'] , prefix='ExterCond' )

full_df = pd.concat([full_df,ExterCondDf],axis=1)

full_df.drop('ExterCond',axis=1,inplace=True)



ExterQualDf = pd.get_dummies( full_df['ExterQual'] , prefix='ExterQual' )

full_df = pd.concat([full_df,ExterQualDf],axis=1)

full_df.drop('ExterQual',axis=1,inplace=True)



FoundationDf = pd.get_dummies( full_df['Foundation'] , prefix='Foundation' )

full_df = pd.concat([full_df,FoundationDf],axis=1)

full_df.drop('Foundation',axis=1,inplace=True)









GarageCondDf = pd.get_dummies( full_df['GarageCond'] , prefix='GarageCond' )

full_df = pd.concat([full_df,GarageCondDf],axis=1)

full_df.drop('GarageCond',axis=1,inplace=True)



GarageFinishDf = pd.get_dummies( full_df['GarageFinish'] , prefix='GarageFinish' )

full_df = pd.concat([full_df,GarageFinishDf],axis=1)

full_df.drop('GarageFinish',axis=1,inplace=True)



GarageQualDf = pd.get_dummies( full_df['GarageQual'] , prefix='GarageQual' )

full_df = pd.concat([full_df,GarageQualDf],axis=1)

full_df.drop('GarageQual',axis=1,inplace=True)



GarageTypeDf = pd.get_dummies( full_df['GarageType'] , prefix='GarageType' )

full_df = pd.concat([full_df,GarageTypeDf],axis=1)

full_df.drop('GarageType',axis=1,inplace=True)



HeatingDf = pd.get_dummies( full_df['Heating'] , prefix='Heating' )

full_df = pd.concat([full_df,HeatingDf],axis=1)

full_df.drop('Heating',axis=1,inplace=True)



HeatingQCDf = pd.get_dummies( full_df['HeatingQC'] , prefix='HeatingQC' )

full_df = pd.concat([full_df,HeatingQCDf],axis=1)

full_df.drop('HeatingQC',axis=1,inplace=True)







HouseStyleDf = pd.get_dummies( full_df['HouseStyle'] , prefix='HouseStyle' )

full_df = pd.concat([full_df,HouseStyleDf],axis=1)

full_df.drop('HouseStyle',axis=1,inplace=True)



LandContourDf = pd.get_dummies( full_df['LandContour'] , prefix='LandContour' )

full_df = pd.concat([full_df,LandContourDf],axis=1)

full_df.drop('LandContour',axis=1,inplace=True)



LandSlopeDf = pd.get_dummies( full_df['LandSlope'] , prefix='LandSlope' )

full_df = pd.concat([full_df,LandSlopeDf],axis=1)

full_df.drop('LandSlope',axis=1,inplace=True)







LotConfigDf = pd.get_dummies( full_df['LotConfig'] , prefix='LotConfig' )

full_df = pd.concat([full_df,LotConfigDf],axis=1)

full_df.drop('LotConfig',axis=1,inplace=True)



LotShapeDf = pd.get_dummies( full_df['LotShape'] , prefix='LotShape' )

full_df = pd.concat([full_df,LotShapeDf],axis=1)

full_df.drop('LotShape',axis=1,inplace=True)



MasVnrTypeDf = pd.get_dummies( full_df['MasVnrType'] , prefix='MasVnrType' )

full_df = pd.concat([full_df,MasVnrTypeDf],axis=1)

full_df.drop('MasVnrType',axis=1,inplace=True)



# Neighborhood     object

# PavedDrive       object

# RoofMatl         object

# RoofStyle        object

# SaleCondition    object

# Street           object



NeighborhoodDf = pd.get_dummies( full_df['Neighborhood'] , prefix='Neighborhood' )

full_df = pd.concat([full_df,NeighborhoodDf],axis=1)

full_df.drop('Neighborhood',axis=1,inplace=True)



PavedDriveDf = pd.get_dummies( full_df['PavedDrive'] , prefix='PavedDrive' )

full_df = pd.concat([full_df,PavedDriveDf],axis=1)

full_df.drop('PavedDrive',axis=1,inplace=True)



RoofMatlDf = pd.get_dummies( full_df['RoofMatl'] , prefix='RoofMatl' )

full_df = pd.concat([full_df,RoofMatlDf],axis=1)

full_df.drop('RoofMatl',axis=1,inplace=True)



RoofStyleDf = pd.get_dummies( full_df['RoofStyle'] , prefix='RoofStyle' )

full_df = pd.concat([full_df,RoofStyleDf],axis=1)

full_df.drop('RoofStyle',axis=1,inplace=True)



SaleConditionDf = pd.get_dummies( full_df['SaleCondition'] , prefix='SaleCondition' )

full_df = pd.concat([full_df,SaleConditionDf],axis=1)

full_df.drop('SaleCondition',axis=1,inplace=True)



StreetDf = pd.get_dummies( full_df['Street'] , prefix='Street' )

full_df = pd.concat([full_df,StreetDf],axis=1)

full_df.drop('Street',axis=1,inplace=True)
# full_X=pd.concat([full_df['OverallQual'],

#                   full_df['GrLivArea'],

#                   full_df['GarageCars'],

#                   full_df['YearBuilt'],

#                   full_df['GarageArea'],

#                   full_df['FullBath'],

#                   full_df['YearRemodAdd'],

#                   full_df['TotRmsAbvGrd'],

#                   full_df['TotalBsmtSF'],

#                   full_df['1stFlrSF']],axis=1)
missing = full_df.isnull().sum()

missing[missing>0].sort_values(ascending=False)
sourceRow=1449

data_X=full_df.loc[0:sourceRow-1,:]

data_y=full_df.loc[0:sourceRow-1,'SalePrice']

pred_X=full_df.loc[sourceRow:,:]



y_df = pd.DataFrame(data_y, columns=['SalePrice'])



y_df['SalePrice'] = np.log(y_df['SalePrice'])



y_df
data_X = data_X.drop(columns=['SalePrice'], axis=1)

data_X
#fit_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge



X_train, X_valid, y_train, y_valid = train_test_split(

                                    data_X, y_df, random_state=0, test_size=.33)

# LRregressor=LinearRegression()

# LRregressor.fit(X_train,y_train)



RF = RandomForestClassifier(n_estimators=100)

RF.fit(X_train,y_train.astype('int'))



XGBRegressor = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGBRegressor.fit(X_train,y_train,

                 early_stopping_rounds=5, 

                 eval_set=[(X_valid, y_valid)],

                 verbose=False)



RidgeRegressor = Ridge(alpha=1.0)

RidgeRegressor.fit(X_train,y_train)



SVRRegressor = SVR(gamma='scale', C=1.0, epsilon=0.2)

SVRRegressor.fit(X_train,y_train)



KernelRidgeRegressor = KernelRidge(alpha=1.0)

KernelRidgeRegressor.fit(X_train,y_train)
# from sklearn.model_selection import cross_val_score



# def rmse_cv(model, X, y):

#     return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
# names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]

# models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),

#           ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

#           ExtraTreesRegressor(),XGBRegressor()]

# for name, model in zip(names, models):

#     score = rmse_cv(model, X_train, y_train)

#     print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
# class grid():

#     def __init__(self,model):

#         self.model = model

    

#     def grid_get(self,X,y,param_grid):

#         grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")

#         grid_search.fit(X,y)

#         print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

#         grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

#         print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
# grid(Lasso()).grid_get(X_train,y_train,{'alpha': [0.0004,0.0005,0.0007,0.0009,0.0011],'max_iter':[10000, 20000]})
# grid(Ridge()).grid_get(X_train,y_train,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
# param_grid={'alpha':[0.2,0.3,0.4], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1]}

# grid(KernelRidge()).grid_get(X_train,y_train,param_grid)
# grid(SVR()).grid_get(X_train,y_train,{'C':[11,13,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
X_valid
#evaluate

# prediction_LR = LRregressor.predict(X_valid)

prediction_RF = RF.predict(X_valid)

prediction_XGB = XGBRegressor.predict(X_valid)

prediction_Ridge = RidgeRegressor.predict(X_valid)

prediction_SVR = SVRRegressor.predict(X_valid)

prediction_KernelRidge = KernelRidgeRegressor.predict(X_valid)



# print ('LR RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_LR)))

print ('RF RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_RF)))

print ('XGB RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_XGB)))

print ('Ridge RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_Ridge)))

print ('SVR RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_SVR)))

print ('KernelRidge RMSE is: \n', np.sqrt(mean_squared_error(y_valid, prediction_KernelRidge)))
pred_X = pred_X.drop(columns=['SalePrice'], axis=1)

pred_X
#predict

predicted_prices = RidgeRegressor.predict(pred_X)
#ensemble
#stack
final_predictions = np.exp(predicted_prices)

final_predictions
print(final_predictions.shape)

result_df = pd.DataFrame(final_predictions, columns=['SalePrice'])

result_df['Id'] = test_df["Id"]

final_df = result_df[['Id', 'SalePrice']]

final_df
#submission

final_df.to_csv('submission1017.csv', index=False)