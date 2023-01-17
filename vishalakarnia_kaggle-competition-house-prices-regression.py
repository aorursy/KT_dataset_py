import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 2000)
h_train=pd.read_csv("../input/house-prices-advanced-regression-techniques")

h_train.head()
h_train.dtypes.head()
h_train.isnull().sum().head()
total_missing=h_train.isnull().sum().sort_values()

percMissing = h_train.isnull().sum() / h_train.isnull().count().sort_values()*100

missing = pd.concat([total_missing, percMissing], axis = 1, keys = ['total #', '%'])

missing[missing['total #'] > 0]
## as we can see there are 4 features having  more than 80% null value it's better to drop that features rather than try to fill them



h_train.drop(["PoolQC","MiscFeature","Fence","Alley"],axis=1,inplace=True)
h_train['SalePrice'].describe()
sns.distplot(h_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % h_train['SalePrice'].skew())

print("Kurtosis: %f" % h_train['SalePrice'].kurt())
#scatter plot GrLivArea/saleprice

var = 'GrLivArea'

data = pd.concat([h_train['SalePrice'], h_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([h_train['SalePrice'], h_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# Relationship with categorical features

sns.barplot(h_train.OverallQual,h_train.SalePrice)
plt.subplots(figsize=(12, 9))

sns.heatmap(h_train.corr())
#'SalePrice' correlation matrix (zoomed heatmap style) take only those columns from upper heatmap

col=h_train[['SalePrice','GarageYrBlt','OverallQual','GarageCars','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt','TotRmsAbvGrd']]

col.corr()

h_train.shape
print("Find most important features relative to target")

corr = h_train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(h_train[cols], size = 2.5)

plt.show();
h_train[['FireplaceQu','LotFrontage','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'BsmtQual','Electrical','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt','MasVnrArea','MasVnrType']].dtypes
h_train.shape
h_train[['LotFrontage','SalePrice']].corr()
sns.scatterplot(x = 'SalePrice', y = 'LotFrontage', data = h_train)
h_train[['GarageYrBlt','SalePrice']].corr()
sns.scatterplot(x = 'SalePrice', y = 'GarageYrBlt', data = h_train)
h_train[['MasVnrArea','SalePrice']].corr()
sns.scatterplot(x = 'SalePrice', y = 'MasVnrArea', data = h_train)
h_train['LotFrontage'].replace(np.nan,h_train.LotFrontage.mean(),inplace=True)
h_train['GarageYrBlt'].replace(np.nan,h_train.GarageYrBlt.mean(),inplace=True)
h_train['MasVnrArea'].replace(np.nan,h_train.MasVnrArea.mean(),inplace=True)
h_train.isnull().sum()
h_train.drop(h_train.loc[h_train['Electrical'].isnull()].index,inplace=True)
h_train['Electrical'].isnull().sum()
h_train.shape
# h_train['Alley'].unique()
# h_train['Alley'].replace(np.nan,'No_alley_access',inplace=True)



# sns.countplot(data=h_train,x='Alley')



# #nan replaced with No alley access as per the Data Dictionary
#BsmtQual

h_train['BsmtQual'].unique()





# as per the Data Dictionary nan stands for "No Basement"

#so,

h_train['BsmtQual'].replace(np.nan,'No_Basement',inplace=True)

sns.countplot(data=h_train,x='BsmtQual')
#BsmtCond

h_train['BsmtCond'].unique()



# as per the Data Dictionary nan stands for "No Basement"

#so,

h_train['BsmtCond'].replace(np.nan,'No_Basement',inplace=True)



sns.countplot(data=h_train,x='BsmtCond')
#BsmtExposure

h_train['BsmtExposure'].unique()



# as per the Data Dictionary nan stands for "No Basement"

#so,

h_train['BsmtExposure'].replace(np.nan,'No_Basement',inplace=True)



sns.countplot(data=h_train,x='BsmtExposure')
#BsmtFinType1

h_train['BsmtFinType1'].unique()



# as per the Data Dictionary nan stands for "No Basement"

#so,

h_train['BsmtFinType1'].replace(np.nan,'No_Basement',inplace=True)



sns.countplot(data=h_train,x='BsmtFinType1')
#BsmtFinType2

h_train['BsmtFinType2'].unique()



# as per the Data Dictionary nan stands for "No Basement"

#so,

h_train['BsmtFinType2'].replace(np.nan,'No_Basement',inplace=True)



sns.countplot(data=h_train,x='BsmtFinType2')
#FireplaceQu

h_train['FireplaceQu'].unique()



# as per the Data Dictionary nan stands for "No Fireplace"

#so,

h_train['FireplaceQu'].replace(np.nan,'No_Fireplace',inplace=True)



sns.countplot(data=h_train,x='FireplaceQu')
#GarageType

h_train['GarageType'].unique()



# as per the Data Dictionary nan stands for "No Garage"

#so,

h_train['GarageType'].replace(np.nan,'No_Garage',inplace=True)



sns.countplot(data=h_train,x='GarageType')
#GarageFinish

h_train['GarageFinish'].unique()



# as per the Data Dictionary nan stands for "No Garage"

#so,

h_train['GarageFinish'].replace(np.nan,'No_Garage',inplace=True)



sns.countplot(data=h_train,x='GarageFinish')

#GarageQual

h_train['GarageQual'].unique()



# as per the Data Dictionary nan stands for "No Garage"

#so,

h_train['GarageQual'].replace(np.nan,'No_Garage',inplace=True)



sns.countplot(data=h_train,x='GarageQual')
#GarageCond

h_train['GarageCond'].unique()



# as per the Data Dictionary nan stands for "No Garage"

#so,

h_train['GarageCond'].replace(np.nan,'No_Garage',inplace=True)



sns.countplot(data=h_train,x='GarageCond')
# #PoolQC

# h_train['PoolQC'].unique()



# # as per the Data Dictionary nan stands for "No Pool"

# #so,

# h_train['PoolQC'].replace(np.nan,'No_Pool',inplace=True)



# sns.countplot(data=h_train,x='PoolQC')
# #Fence

# h_train['Fence'].unique()



# # as per the Data Dictionary nan stands for "No Fence"

# #so,

# h_train['Fence'].replace(np.nan,'No_Fence',inplace=True)



# sns.countplot(data=h_train,x='Fence')

# #MiscFeature

# h_train['MiscFeature'].unique()



# # as per the Data Dictionary nan stands for "None"

# #so,

# h_train['MiscFeature'].replace(np.nan,'None',inplace=True)



# sns.countplot(data=h_train,x='MiscFeature')
h_train.shape
#MasVnrType

h_train['MasVnrType'].unique()



#in this there is no designation for nan so we are removing the nan values

h_train.drop(h_train.loc[h_train['MasVnrType'].isnull()].index,inplace=True)
sns.heatmap(h_train.isnull())
# #standardizing data

# saleprice_scaled = StandardScaler().fit_transform(h_train['SalePrice'][:,np.newaxis]);

# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

# high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

# print('outer range (low) of the distribution:')

# print(low_range)

# print('\nouter range (high) of the distribution:')

# print(high_range)
# sns.distplot(h_train['SalePrice'], fit=norm);

# fig = plt.figure()

# res = stats.probplot(h_train['SalePrice'], plot=plt)
# h_train['SalePrice'].quantile([0.1,0.2,0.3,0.4])

# ### as we can see from above graph there is one outlier at -3 std and 2 at +3 std and same we can see below also

# h_train['SalePrice'].quantile([0.97,0.98,0.99,1])
# #applying log transformation

# h_train['SalePrice'] = np.log(h_train['SalePrice'])
# #transformed histogram and normal probability plot

# sns.distplot(h_train['SalePrice'], fit=norm);

# fig = plt.figure()

# res = stats.probplot(h_train['SalePrice'], plot=plt)
# h_train['SalePrice'].quantile([0.1,0.2,0.3,0.4])
# h_train['SalePrice'].quantile([0.97,0.98,0.99,1])
# h_train.drop(h_train[h_train['SalePrice']<11.728037].index,axis=0,inplace=True)
# h_train.drop(h_train[h_train['SalePrice']>12.993142].index,axis=0,inplace=True)
# h_train=h_train.drop('Id',axis=1)
# h_train.shape
# #LotFrontage

# sns.distplot(h_train['LotFrontage'], fit=norm);

# fig = plt.figure()

# res = stats.probplot(h_train['LotFrontage'], plot=plt)
# h_train['LotFrontage'].quantile([0.1,0.2,0.3,0.4])
# h_train['LotFrontage'].quantile([0.96,0.97,0.98,0.99,1])
# h_train.drop(h_train[h_train['LotFrontage']>139.2].index,axis=0,inplace=True)
# h_train.shape
# #LotArea

# sns.distplot(h_train['GrLivArea'], fit=norm);

# fig = plt.figure()

# res = stats.probplot(h_train['GrLivArea'], plot=plt)
# h_train['GrLivArea'].quantile([0.1,0.2,0.3,0.4])
# h_train['GrLivArea'].quantile([0.97,0.98,0.99,1])
# h_train.drop(h_train[h_train['GrLivArea']>2931.84].index,axis=0,inplace=True)
# h_train.shape
# sns.distplot(h_train['TotalBsmtSF'], fit=norm);

# fig = plt.figure()

# res = stats.probplot(h_train['TotalBsmtSF'], plot=plt)
# h_train['TotalBsmtSF'].quantile([0.1,0.2,0.3,0.4])
# h_train['TotalBsmtSF'].quantile([0.97,0.98,0.99,1])
# h_train.drop(h_train[h_train['TotalBsmtSF']<814.0].index,axis=0,inplace=True)
# h_train.drop(h_train[h_train['TotalBsmtSF']>2077.84].index,axis=0,inplace=True)
# h_train.shape
h_train.drop('Id',axis=1,inplace=True)
h_train.corr()
h1_train=h_train[["SalePrice","OverallQual","YearBuilt","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","GarageCars","GarageArea","TotRmsAbvGrd"]]
# These are the best correlation with saleprice

# OverallQual      0.790982

# GrLivArea        0.708624

# GarageCars       0.640409

# GarageArea       0.623431

# TotalBsmtSF      0.613581

# 1stFlrSF         0.605852

# FullBath         0.560664

# TotRmsAbvGrd     0.533723

# YearBuilt        0.522897
h1_train.shape
h1_train['TotRmsAbvGrd'].dtype
h1_train_dum=pd.get_dummies(h1_train,drop_first=True)
h1_train_dum.shape
x=h1_train_dum.drop(['SalePrice'],axis=1)

y=h1_train_dum['SalePrice']
x.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y , test_size=0.2 , random_state=21 )
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
clf = RandomForestRegressor()



param_dist = {"n_estimators": [50, 100, 150,200]}



clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))



print('*'*20)





print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('*'*20)





r2_score=metrics.r2_score(y_test,y_pred)

print('r2_score:',r2_score)
from sklearn.linear_model import LinearRegression
clf_lr=LinearRegression()



clf_lr.fit(x_train,y_train)
y_pred_lr=clf_lr.predict(x_test)
print('MAE:',metrics.mean_absolute_error(y_test,y_pred_lr))



print('*'*20)





print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lr)))

print('*'*20)





r2_score=metrics.r2_score(y_test,y_pred_lr)

print('r2_score:',r2_score)
from sklearn.model_selection import cross_val_score,KFold
kf=KFold(n_splits=5)

RFRegressor=RandomForestRegressor(random_state=5)



score=cross_val_score(RFRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')



r=score.mean()

print(r)
from math import sqrt



sqrt(-r)
## use this

kf=KFold(n_splits=5)

LRegressor=LinearRegression()



score=cross_val_score(LRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')



r=score.mean()

print(r)
from math import sqrt



sqrt(-r)
import xgboost as xgb
model = xgb.XGBRegressor()



model.fit(x_train,y_train)
y_pred_xgb=model.predict(x_test)



y_pred_xgb
print('MAE:',metrics.mean_absolute_error(y_test,y_pred_xgb))



print('*'*20)





print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_xgb)))

print('*'*20)





r2_score=metrics.r2_score(y_test,y_pred_xgb)

print('r2_score:',r2_score)
kf=KFold(n_splits=5)

xgbRegressor=xgb.XGBRegressor()



score=cross_val_score(xgbRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')



r=score.mean()

print(r)
from math import sqrt



sqrt(-r)
h_test=pd.read_csv("test.csv")

h_test.head()
h_test.shape
total_missing_t=h_test.isnull().sum().sort_values()

percMissing_t = h_test.isnull().sum() / h_test.isnull().count().sort_values()*100

missing_t = pd.concat([total_missing_t, percMissing_t], axis = 1, keys = ['total #', '%'])

missing_t[missing_t['total #'] > 0]
# h_test['Alley'].replace(np.nan,'No_alley_access',inplace=True)



# #sns.countplot(data=h_train,x='Alley')
#BsmtCond

# h_test['BsmtCond'].replace(np.nan,'No_Basement',inplace=True)
 #BsmtExposure

# h_test['BsmtExposure'].replace(np.nan,'No_Basement',inplace=True)
h1_test=h_test[["OverallQual","YearBuilt","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","GarageCars","GarageArea","TotRmsAbvGrd"]]
h1_test.isnull().sum()
#GarageCars

h1_test.dtypes

# h_test['GarageCars'].replace(np.nan,'No_Basement',inplace=True)
h1_test.drop(h1_test.loc[h1_test['GarageCars'].isnull()].index,inplace=True)
h1_test.drop(h1_test.loc[h1_test['TotalBsmtSF'].isnull()].index,inplace=True)
h1_test_dum=pd.get_dummies(h1_test,drop_first= True)
y_pred_xgb_test=model.predict(h1_test_dum)
y_pred_xgb_test=pd.DataFrame(y_pred_xgb_test)
y_pred_xgb_test.head()
sample=pd.read_csv('sample_submission.csv')
sample.head()
submit=pd.concat([sample.Id,y_pred_xgb_test],axis=1)
submit.head()
submit.columns=["Id","SalePrice"]
# sns.lmplot("Id","SalePrice",data=submit,fit_reg=True)
submit.to_csv("Submission_HLP_kaggle.csv",index=False)
submit.shape
submit.loc[submit["SalePrice"].isnull()]
from sklearn.model_selection import RandomizedSearchCV

from numpy import nan
import xgboost as xgb

model = xgb.XGBRegressor()



model.fit(x_train,y_train)
Booster=["gbtree","gblinear"]

base_score=[0.25,0.50,0.75,1]
n_estimators=[100,500,900,1000,1500]

max_depth=[2,3,5,10,15]

Booster=["gbtree","gblinear"]

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



hyperparameter_grid={

    "n_estimators":n_estimators,

    "max_depth":max_depth,

    "Booster":Booster,

    "learning_rate":learning_rate,

    "min_child_weight":min_child_weight,

    "base_score":base_score

    

}

random_cv=RandomizedSearchCV(estimator=model,

                            param_distributions=hyperparameter_grid,

                            cv=5,n_iter=50,

                            scoring="neg_mean_absolute_error",n_jobs=4,

                            verbose=5,

                            return_train_score=True,

                            random_state=42)
random_cv.fit(x_train,y_train)
random_cv.best_estimator_
regressor=xgb.XGBRegressor(Booster='gbtree', base_score=0.5, booster=None,

             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,

             gamma=0, gpu_id=-1, importance_type='gain',

             interaction_constraints=None, learning_rate=0.15, max_delta_step=0,

             max_depth=2, min_child_weight=2, missing=nan,

             monotone_constraints=None, n_estimators=100, n_jobs=0,

             num_parallel_tree=1, objective='reg:squarederror', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method=None, validate_parameters=False, verbosity=None)
regressor.fit(x_train,y_train)
y_pred_ran=regressor.predict(h1_test_dum)
y_pred_ran
y_pred_ran=pd.DataFrame(y_pred_ran)
submit_ran=pd.concat([sample.Id,y_pred_ran],axis=1)
submit_ran.head()
submit_ran.columns=["Id","SalePrice"]
submit_ran.to_csv("Submission_HLP_ran_kaggle.csv",index=False)