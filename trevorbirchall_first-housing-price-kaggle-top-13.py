import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from sklearn.metrics import confusion_matrix

from scipy.stats import norm

import pandas as pd

import numpy as np

import math 

import xgboost as xgb

np.random.seed(1993)

from scipy.stats import skew

from scipy import stats



import statsmodels

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

%matplotlib inline

print("done")
import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

pd.set_option('display.max_columns', None)
train.head()
train_ID = train['Id']

test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True, figsize=(20,20))

fig.tight_layout()



Attributes = [("LotFrontage", train['LotFrontage']) , ("GrLivArea",train['GrLivArea']), ("LotArea",train['LotArea']), ("MasVnrArea",train['MasVnrArea']), \

             ("BsmtFinSF1", train['BsmtFinSF1']) , ("BsmtFinSF2",train['BsmtFinSF2']), ("TotalBsmtSF",train['TotalBsmtSF']), ("1stFlrSF",train['1stFlrSF']), \

             ("LowQualFinSF", train['LowQualFinSF']), ("GarageCars1", train['GarageCars']), ("GarageArea",train['GarageArea']), ("WoodDeckSF",train['WoodDeckSF']), \

             ("OpenPorchSF",train['OpenPorchSF']), ("EnclosedPorch", train['EnclosedPorch']), ("3SsnPorch",train['3SsnPorch']), ("ScreenPorch",train['ScreenPorch']), \

             ("PoolArea",train['PoolArea']), ("MiscVal",train['MiscVal'])]



nclassifier = 0

for ax in axes.flat:

    name = Attributes[nclassifier][0]

    classifier = Attributes[nclassifier][1]

    ax.scatter(x = classifier, y = train['SalePrice'])

    ax.set_title(name)

    nclassifier += 1
train[train.LotFrontage > 200]
train[(train.GrLivArea > 4000) & (train.SalePrice < 300000)]
train[train.LotArea > 100000]
train[train.MasVnrArea > 1400]
train[train.BsmtFinSF1> 2500]
train[train.BsmtFinSF2> 1200]
train[train.TotalBsmtSF> 3500]
train[train["1stFlrSF"] > 3500]
train[train.EnclosedPorch > 400]
train[train.MiscVal> 4000]
train = train.drop([train.index[934], train.index[1298], train.index[523], train.index[249], train.index[313],  

                    train.index[335], train.index[706], train.index[297], train.index[322], train.index[197],

                    train.index[346], train.index[1230]])
fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True, figsize=(20,20))

fig.tight_layout()



Attributes = [("LotFrontage", train['LotFrontage']) , ("GrLivArea",train['GrLivArea']), ("LotArea",train['LotArea']), ("MasVnrArea",train['MasVnrArea']), \

             ("BsmtFinSF1", train['BsmtFinSF1']) , ("BsmtFinSF2",train['BsmtFinSF2']), ("TotalBsmtSF",train['TotalBsmtSF']), ("1stFlrSF",train['1stFlrSF']), \

             ("LowQualFinSF", train['LowQualFinSF']), ("GarageCars1", train['GarageCars']), ("GarageArea",train['GarageArea']), ("WoodDeckSF",train['WoodDeckSF']), \

             ("OpenPorchSF",train['OpenPorchSF']), ("EnclosedPorch", train['EnclosedPorch']), ("3SsnPorch",train['3SsnPorch']), ("ScreenPorch",train['ScreenPorch']), \

             ("PoolArea",train['PoolArea']), ("MiscVal",train['MiscVal'])]



nclassifier = 0

for ax in axes.flat:

    name = Attributes[nclassifier][0]

    classifier = Attributes[nclassifier][1]

    ax.scatter(x = classifier, y = train['SalePrice'])

    ax.set_title(name)

    nclassifier += 1
g = sns.distplot(train["SalePrice"], fit=norm, label="Skewness : %.3f"%(train["SalePrice"].skew()))

g = g.legend(loc="best")
train["SalePrice"] = train["SalePrice"].map(lambda i: np.log(i) if i > 0 else 0)
train['SalePrice'].head()
g = sns.distplot(train["SalePrice"], fit=norm, label="Skewness : %.3f"%(train["SalePrice"].skew()))

g = g.legend(loc="best")
ntrain = train.shape[0]

ntest = test.shape[0]

trainY = train.SalePrice

data = pd.concat((train, test)).reset_index(drop=True)

data.drop(["SalePrice"], axis = 1, inplace = True)
data.describe()
data.shape
corrmat = data.corr()

plt.subplots(figsize=(15,15))

sns.heatmap(corrmat, vmax=0.8, square=True)
data_na = data.isnull().sum()

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

data_na_Per = (data.isnull().sum() / len(data)) * 100

data_na_Per = data_na_Per.drop(data_na_Per[data_na_Per == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Sum' :data_na, "Missing Ratio" : data_na_Per})

missing_data.head(30)
def filling_missing_values(data,variable, new_value):

    data[variable] = data[variable].fillna(new_value)
filling_missing_values(data,'PoolQC', "None")

filling_missing_values(data,'MiscFeature', "None")

filling_missing_values(data,'Alley', "None")

filling_missing_values(data,'Fence', "None")

filling_missing_values(data,'FireplaceQu', "None")

filling_missing_values(data,'GarageType', "None")

filling_missing_values(data,'GarageFinish', "None")

filling_missing_values(data,'GarageQual', "None")

filling_missing_values(data,'GarageCond', "None")

filling_missing_values(data,'GarageYrBlt', 0)

filling_missing_values(data,'GarageArea', 0)

filling_missing_values(data,'GarageCars', 0)

filling_missing_values(data,'BsmtCond', "None")

filling_missing_values(data,'BsmtQual', "None")

filling_missing_values(data,'BsmtExposure', "None")

filling_missing_values(data,'BsmtFinType1', "None") 

filling_missing_values(data,'BsmtFinType2', "None")

filling_missing_values(data,'BsmtFinSF1', 0)

filling_missing_values(data,'BsmtFinSF2', 0)

filling_missing_values(data,'BsmtUnfSF', 0)

filling_missing_values(data,'TotalBsmtSF', 0)

filling_missing_values(data,'BsmtFullBath', 0)

filling_missing_values(data,'BsmtHalfBath', 0)
data.MasVnrType.value_counts()
g, ax = plt.subplots(figsize=(15, 6))

fig = sns.barplot(x="Neighborhood", y="LotFrontage", data=data)
g, ax = plt.subplots(figsize=(15, 6))

fig = sns.barplot(x="BldgType", y="LotFrontage", data=data)
data["LotFrontage"] = data.groupby(["Neighborhood","BldgType"])["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

data["LotFrontage"] = data.groupby("BldgType")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
g, ax = plt.subplots(figsize=(8, 6))

fig = sns.countplot(x="MasVnrType", data=data)
filling_missing_values(data,'MasVnrType', "None")

filling_missing_values(data,'MasVnrArea', 0)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data["Functional"] = data["Functional"].fillna(data["Functional"].mode()[0])

data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

data["KitchenQual"] = data["KitchenQual"].fillna(data["KitchenQual"].mode()[0])

data["Exterior1st"] = data["Exterior1st"].fillna(data["Exterior1st"].mode()[0])

data["Exterior2nd"] = data["Exterior2nd"].fillna(data["Exterior2nd"].mode()[0])

data["SaleType"] = data["SaleType"].fillna(data["SaleType"].mode()[0])
data.groupby('Utilities').Utilities.value_counts()
data=data.drop(["Utilities"], axis=1)
data_na = data.isnull().sum()

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

data_na_Per = (data.isnull().sum() / len(data)) * 100

data_na_Per = data_na_Per.drop(data_na_Per[data_na_Per == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Sum' :data_na, "Missing Ratio" : data_na_Per})

missing_data.head(30)
T = data.columns.to_series().groupby(data.dtypes).groups

T
data['MSSubClass'] = data['MSSubClass'].astype(str)

data['OverallCond'] = data['OverallCond'].astype(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]

data["Bathrooms"] = data['FullBath'] + data['HalfBath']*0.5 + data['BsmtFullBath'] + data['BsmtHalfBath']*0.5
data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))

    

print('Shape data: {}'.format(data.shape))
data.dtypes.value_counts()
numeric_feats = data.dtypes[data.dtypes != "object"].index



skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head()
skewness = skewness[abs(skewness.Skew)>0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    data[feat] = boxcox1p(data[feat], lam)
data = pd.get_dummies(data)

data.shape
data.head()
from sklearn.model_selection import train_test_split

trainX = data[:ntrain]

test = data[ntrain:]
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import  KFold, cross_val_score, KFold,RandomizedSearchCV, learning_curve, GridSearchCV

from xgboost.sklearn import XGBClassifier

import xgboost as xgb

import lightgbm as lgb

import math

import sklearn.model_selection as ms

import sklearn.metrics as sklm



from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error

from matplotlib.legend_handler import HandlerLine2D
from sklearn.preprocessing import RobustScaler

scaler= RobustScaler()



trainX = scaler.fit_transform(trainX)

TestX= scaler.transform(test)
Results = pd.DataFrame({'Model': [],'RMSE Train': []})
Linear_mod = LinearRegression()

Linear_mod.fit(trainX,trainY)

y_pred_train=Linear_mod.predict(trainX)

res = pd.DataFrame({"Model":['Linear Regression'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
alpha = np.arange(10.0,20.001,0.01)



param_gridRidge = dict(alpha = alpha)



print(param_gridRidge)
'''

grid_ridge=ms.GridSearchCV(Ridge(), param_gridRidge, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_ridge.fit(trainX, trainY)

print("The best parameters are: ",grid_ridge.best_params_)

print("The mean square error is: ", -grid_ridge.best_score_)

print("RMSE: ", str(math.sqrt(-grid_ridge.best_score_)))

'''
ridge_mod = Ridge(alpha = 10)

ridge_mod.fit(trainX,trainY)

y_pred_train=ridge_mod.predict(trainX)

res = pd.DataFrame({"Model":['Ridge'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train)))) 
alpha = np.arange(1.0,3.01,0.05)

kernel = ["polynomial"]

degree = np.arange(1,5,0.5)

coef = np.arange(1,5,0.5)



param_gridKRidge = dict(alpha = alpha, kernel = kernel, degree = degree, coef0=coef)



print(param_gridKRidge)
'''

grid_KRidge=ms.GridSearchCV(KernelRidge(), param_gridKRidge, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_KRidge.fit(trainX, trainY)

print("The best parameters are: ",grid_KRidge.best_params_)

print("The mean square error is: ", -grid_KRidge.best_score_)

print("RMSE: ", str(math.sqrt(-grid_KRidge.best_score_)))

'''
KRidge_mod = KernelRidge(alpha = 1.0, coef0 = 4.5, degree = 2.0, kernel = "polynomial")

KRidge_mod.fit(trainX,trainY)

y_pred_train=KRidge_mod.predict(trainX)

res = pd.DataFrame({"Model":['Kernel Ridge'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))  
alpha = (0.0001,0.0005,0.00075,0.001,0.005,0.01,0.05,0.1,0.5,0,1,10,50,100)



param_gridLasso = dict(alpha = alpha)



print(param_gridLasso)
'''

grid_Lasso=ms.GridSearchCV(Lasso(), param_gridLasso, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_Lasso.fit(trainX, trainY)

print("The best parameters are: ",grid_Lasso.best_params_)

print("The mean square error is: ", -grid_Lasso.best_score_)

print("RMSE: ", str(math.sqrt(-grid_Lasso.best_score_)))

'''
Lasso_mod = Lasso(0.0005)

Lasso_mod.fit(trainX,trainY)

y_pred_train=Lasso_mod.predict(trainX)

res = pd.DataFrame({"Model":['Lasso'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
alpha = (0.0001,0.0005,0.00075,0.001,0.005,0.01,0.05,0.1,0.5,0,1,10)

l1_ratio = np.arange(0.1, 1.01, 0.05)



param_gridENet = dict(alpha = alpha, l1_ratio = l1_ratio)



print(param_gridENet)
'''

grid_ENet=ms.GridSearchCV(ElasticNet(), param_gridENet, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_ENet.fit(trainX, trainY)

print("The best parameters are: ",grid_ENet.best_params_)

print("The mean square error is: ", -grid_ENet.best_score_)

print("RMSE: ", str(math.sqrt(-grid_ENet.best_score_)))

'''
ENet_mod = ElasticNet(alpha = 0.0005, l1_ratio = 0.65)

ENet_mod.fit(trainX,trainY)

y_pred_train=ENet_mod.predict(trainX)

res = pd.DataFrame({"Model":['ENet'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
learning_rates = (0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)

n_estimators = (100,500,1000,1500,2000,2500,3000,3500)

min_samples_split = [2, 5, 10, 15, 25]

min_samples_leaf = [1, 2, 4, 10, 15, 25]

max_depth = (4, 6, 10)



param_gridGB = dict(learning_rate = learning_rates, n_estimators = n_estimators, min_samples_split= min_samples_split, 

                    min_samples_leaf = min_samples_leaf, max_depth = max_depth)



print(param_gridGB)
'''

grid_GB=ms.GridSearchCV(GradientBoostingRegressor(), param_gridGB, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_GB.fit(trainX, trainY)

print("The best parameters are: ",grid_GB.best_params_)

print("The mean square error is: ", -grid_GB.best_score_)

print("RMSE: ", str(math.sqrt(-grid_GB.best_score_)))

'''
## grid_GB.best_estimator_

GB_mod = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, loss='huber')



GB_mod.fit(trainX,trainY)

y_pred_train=GB_mod.predict(trainX)

res = pd.DataFrame({"Model":['Gradient Boosting'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
learning_rates = (0.01, 0.05, 0.1, 0.5, 1)

n_estimators = (1500,2000,2500,3000,3460)

min_child_weights = (0, 1, 1.25, 2)

gammas = (0, 0.5, 1)

subsamples = (0.0, 0.25, 0.5, 0.7, 1.0)

colsample_bytrees = (0.0, 0.25, 0.5, 0.7, 1.0)

max_depth = (3, 4, 5)

reg_apha = (0, 0.1, 0.25, 0.5)

reg_lamba = (0, 0.25, 0.5,0.75, 1)



param_gridXGB = dict(max_depth= max_depth, n_estimators=n_estimators, learning_rate=learning_rates, min_child_weight=min_child_weights, 

                     gamma=gammas, subsample=subsamples, colsample_bytree=colsample_bytrees, reg_apha = reg_apha,reg_lamba = reg_lamba)



print(param_gridXGB)
'''

grid_XGB=ms.GridSearchCV(xgb.XGBRegressor(), param_gridXGB, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_XGB.fit(trainX, trainY)

print("The best parameters are: ",grid_XGB.best_params_)

print("The mean square error is: ", -grid_XGB.best_score_)

print("RMSE: ", str(math.sqrt(-grid_XGB.best_score_)))

'''
## grid_XGB.best_estimator_



XGB_mod = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,

                           colsample_bytree=0.7,objective='reg:linear',silent=1, nthread=-1,scale_pos_weight=1,reg_alpha=0.00006)

    

XGB_mod.fit(trainX,trainY)

y_pred_train=XGB_mod.predict(trainX)

res = pd.DataFrame({"Model":['XGB'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
learning_rates = (0.01, 0.05, 0.1, 0.5, 1)

n_estimators = (100,500,720,750,1000,1500,2000,2500,3000)

num_leaves = (2,5,10)

max_bin = (25,50,55,100)

bagging_fraction = (0.1,0.25,0.5,0.75,0.8,1)

bagging_freq = (0,1,5,10)

feature_fraction = np.arange(0.1, 1.01, 0.1)

min_data_in_leaf = (5,6,10,15,20)



param_gridLGBM = dict(learning_rates=learning_rates, n_estimators=n_estimators, num_leaves=num_leaves, 

                     max_bin=max_bin, bagging_fraction=bagging_fraction, bagging_freq=bagging_freq, feature_fraction = feature_fraction, 

                     rmin_data_in_leaf = min_data_in_leaf)



print(param_gridLGBM)
'''

grid_LGBM=ms.GridSearchCV(lgb.LGBMRegressor(), param_gridLGBM, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)

grid_LGBM.fit(trainX, trainY)

print("The best parameters are: ",grid_LGBM.best_params_)

print("The mean square error is: ", -grid_LGBM.best_score_)

print("RMSE: ", str(math.sqrt(-grid_LGBM.best_score_)))

'''
## grid_LGBM.best_estimator_



LGBM_mod = lgb.LGBMRegressor(num_leaves=5, learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



LGBM_mod.fit(trainX,trainY)

y_pred_train=LGBM_mod.predict(trainX)

res = pd.DataFrame({"Model":['LightGBM'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
Results.sort_values(by=['RMSE Train'])
from mlxtend.regressor import StackingCVRegressor

Stack_mod = StackingCVRegressor(regressors=( KRidge_mod, GB_mod, XGB_mod, LGBM_mod), 

                          meta_regressor = GB_mod, use_features_in_secondary=True)

Stack_mod.fit(trainX,trainY)

y_pred_train=Stack_mod.predict(trainX)

res = pd.DataFrame({"Model":['Stacked'],

                    "RMSE Train": [math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))]})



Results = Results.append(res)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(trainY, y_pred_train))))
Results.sort_values(by=['RMSE Train'])
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
Stack_mod.fit(trainX, trainY)

Stacked_train_pred = Stack_mod.predict(trainX)

Stacked_pred = np.expm1(Stack_mod.predict(TestX))
LGBM_mod.fit(trainX, trainY)

LGBM_train_pred = LGBM_mod.predict(trainX)

LGBM_pred = np.expm1(LGBM_mod.predict(TestX))
GB_mod.fit(trainX, trainY)

GB_train_pred = GB_mod.predict(trainX)

GB_pred = np.expm1(GB_mod.predict(TestX))
XGB_mod.fit(trainX, trainY)

XGB_train_pred = XGB_mod.predict(trainX)

XGB_pred = np.expm1(XGB_mod.predict(TestX))
print(rmsle(trainY, GB_train_pred*0.6 + Stacked_train_pred*0.3 + LGBM_train_pred*0.05 + XGB_train_pred*0.05))
ensemble_model = (GB_pred*.6 + Stacked_pred*.3 + LGBM_pred*.05 + XGB_pred*.05)
submission = pd.DataFrame({

        "Id": test_ID,

        "SalePrice": ensemble_model})



submission.to_csv("final_submission.csv", index=False)

submission.head()