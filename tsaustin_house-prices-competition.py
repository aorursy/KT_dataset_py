import pandas as pd

import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

sns.set(rc={'figure.figsize':(10,6)},style="whitegrid")



plt.subplot(1, 2, 1)

plt.scatter(train['LotArea'],train['SalePrice'])

plt.ylabel('Selling Price')

plt.xlabel('Lot Area (SqF)')

plt.title('Sellingprice / Lot Area')

plt.subplot(1, 2, 2)

plt.scatter(train['GrLivArea'],train['SalePrice'])

plt.ylabel('Selling Price')

plt.xlabel('Living Area (SqF)')

plt.title('Sellingprice / Above ground living area')

plt.tight_layout()

plt.show()
train = train[train['LotArea'] < 150000]

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice'] < 200000)].index)
train.shape
# copy the prediction value to a separate vector

target = train['SalePrice']
# plot distribution

from scipy.stats import norm, skew, probplot #for some statistics

sns.set(rc={'figure.figsize':(6,6)},style="whitegrid")



sns.distplot(target , fit=norm);

(mu, sigma) = norm.fit(target)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.show()



# and probability plot

fig = plt.figure()

res = probplot(target, plot=plt)

plt.show()
import numpy as np

target = np.log1p(target)
# plot distribution again

sns.distplot(target , fit=norm);

(mu, sigma) = norm.fit(target)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.show()



# and probability plot

fig = plt.figure()

res = probplot(target, plot=plt)

plt.show()
# before we combine we'll need to store how many samples belong to the training samples

ntrain = train.shape[0]
alldata = pd.concat((train,test)).reset_index(drop=True)

# drop the target variable from the combined data set

alldata.drop(['SalePrice'],axis=1,inplace=True)

alldata.head()
alldata.dtypes
alldata_na = (alldata.isnull().sum() / len(alldata)) * 100

alldata_na = alldata_na.drop(alldata_na[alldata_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :alldata_na})

missing_data.head(20)
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):

    alldata[col] = alldata[col].fillna('None')
alldata['LotFrontage'].fillna(value=0.0,inplace=True)
for col in ('GarageYrBlt', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType'):

    alldata[col] = alldata[col].fillna('None')
alldata['MasVnrArea'].fillna(value=0.0,inplace=True)
for col in ('MSZoning', 'Utilities', 'Functional', 'Exterior2nd', 'Exterior1st', 'Electrical', 'KitchenQual','SaleType'):

    alldata[col] = alldata[col].fillna('None')
for col in ('BsmtHalfBath', 'BsmtFullBath', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1'):

    alldata[col] = alldata[col].fillna(0.0)
alldata.columns
for col in ('YrSold', 'YearRemodAdd', 'YearBuilt', 'OverallCond','OverallQual', 'MSSubClass'):

    alldata[col] = alldata[col].astype(str)
# it think the four features for bathrooms are overkill, let's combine this to one feature

alldata['Bathrooms'] = alldata['BsmtFullBath'] + (0.5*alldata['BsmtHalfBath']) + alldata['FullBath'] + (0.5*alldata['HalfBath'])

alldata = alldata.drop(columns=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
# I think it's enough to use the squarefeet of finished and unfinished basement. Drop the other columns

alldata = alldata.drop(columns=['BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2'], axis=1) 
# drop the pool quality column. we still have the pool area feature

alldata = alldata.drop(columns=['PoolQC'], axis=1)
# find skewed features

numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(20)
# and box cox transform the features with skewness

skewness = skewness[abs(skewness) > 0.75]



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    alldata[feat] = boxcox1p(alldata[feat], lam)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 

        'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'YearRemodAdd', 'OverallQual', 'YearBuilt')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(alldata[c].values)) 

    alldata[c] = lbl.transform(list(alldata[c].values))
alldata = pd.get_dummies(alldata)
alldata.shape
train = alldata[:ntrain]

submission_features = alldata[ntrain:]
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, target, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge



def model_ridge(**kwargs):

    return make_pipeline(RobustScaler(),Ridge(**kwargs))

from sklearn.model_selection import validation_curve



param_range = [0.5,1,2,3,4,5,10,15,20,30]

train_scores, valid_scores = validation_curve(model_ridge(), train.values, target, "ridge__alpha",param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
linridge = model_ridge(alpha=10)

rmsle_cv(linridge).mean()
from sklearn.linear_model import Lasso



def model_lasso(**kwargs):

    return make_pipeline(RobustScaler(),Lasso(**kwargs))
param_range = [0.0001,0.0002, 0.0005,0.001,0.01]

train_scores, valid_scores = validation_curve(model_lasso(), train.values, target, "lasso__alpha"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
linlasso = model_lasso(alpha=0.001)

rmsle_cv(linlasso).mean()
from sklearn.ensemble import GradientBoostingRegressor
param_range = [5,10,20,40,50,100]

train_scores, valid_scores = validation_curve(GradientBoostingRegressor(), train.values, target

                                              , "n_estimators"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
param_range = [0.01,0.05,0.1,0.2]

train_scores, valid_scores = validation_curve(GradientBoostingRegressor(n_estimators=40)

                                              , train.values, target, "learning_rate"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
param_range = [2,3,4,5,6]

train_scores, valid_scores = validation_curve(GradientBoostingRegressor(n_estimators=40,learning_rate=0.1)

                                              , train.values, target, "max_depth"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
gbdt = GradientBoostingRegressor(n_estimators=40,learning_rate=0.1,max_depth=3)

rmsle_cv(gbdt).mean()
import xgboost as xgb
param_range = [40,50,60,70,80,90, 100]

train_scores, valid_scores = validation_curve(xgb.XGBRegressor(), train.values, target

                                              , "n_estimators"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
param_range = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]

train_scores, valid_scores = validation_curve(xgb.XGBRegressor(n_estimators=70), train.values, target

                                              , "learning_rate"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
param_range = [1,2,3,4,5]

train_scores, valid_scores = validation_curve(xgb.XGBRegressor(n_estimators=70, learning_rate=0.2), train.values

                                              , target

                                              , "max_depth"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
xgboost = xgb.XGBRegressor(n_estimators=70, learning_rate=0.2, max_depth=2)

rmsle_cv(xgboost).mean()
from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def neg_rmsle(y, y_pred):

    return -1*np.sqrt(mean_squared_error(y, y_pred))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from mlxtend.regressor import StackingCVRegressor



RANDOM_SEED = 42 



ridge = model_ridge(alpha=10)

lasso = model_lasso(alpha=0.001)

rf = Ridge()



np.random.seed(RANDOM_SEED)

stack = StackingCVRegressor(regressors=[lasso, ridge],

                            meta_regressor=rf, 

                            use_features_in_secondary=False)



param_range = [0.001,0.01,0.1,0.2]

train_scores, valid_scores = validation_curve(stack, train.values

                                              , target

                                              , "meta_regressor__alpha"

                                              ,param_range,cv=3

                                              ,scoring="neg_mean_squared_error", n_jobs=-1)



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

#plt.ylim(0,1)

plt.show()
ridge = model_ridge(alpha=10)

lasso = model_lasso(alpha=0.001)

rf = Ridge(alpha=0.1)



np.random.seed(RANDOM_SEED)

stack = StackingCVRegressor(regressors=[lasso, ridge],

                            meta_regressor=rf, 

                            use_features_in_secondary=False)



stack.fit(train, target)

stack_train_pred = stack.predict(train)

stack_test_pred = np.expm1(stack.predict(submission_features))

print("Stacked rmsle {0}".format(rmsle(target, stack_train_pred)))
submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = stack_test_pred
submission.head()
submission.to_csv('submission.csv',index=False)