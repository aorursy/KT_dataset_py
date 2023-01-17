# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')



df_test = pd.read_csv('../input/test.csv')

df_train.drop('Id',axis=1,inplace=True )

id_test = df_test['Id']                      # for submissions

df_test.drop('Id',axis=1,inplace=True )

df_train.head(5)
df_train.columns
row_train=df_train.shape[0]

row_test=df_test.shape[0]
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
# for numeric variable

df_train.describe().transpose()
# for categrocial variables

df_train.describe(include = ['O']).transpose()
cols = df_train.select_dtypes([np.number]).columns

print(cols)
num_df=df_train[cols]

num_df.head(5)
corrmat = num_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, vmin=0.4, square=True);
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
outliers_GrLivArea = df_train.loc[(df_train['GrLivArea']>4500.0)]

outliers_GrLivArea[['OverallQual','GrLivArea' , 'SalePrice']]

outliers=[524,1299]

df_train = df_train.drop(df_train.index[outliers])

df_train1=df_train[['MSSubClass','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars', 

          'GarageArea','MSSubClass','MSZoning', 'Neighborhood','LotConfig',

'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageType','OverallQual']]
total = df_train1.isnull().sum().sort_values(ascending=False)

percent = (df_train1.isnull().sum()/df_train1.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#fill train na

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',

            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"

           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:

    df_train[col] = df_train[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'

           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):

    df_train[col] = df_train[col].fillna(0)



# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

df_train['LotFrontage'] = df_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# fill test set na

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',

            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"

           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:

    df_test[col] = df_test[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'

           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):

    df_test[col] = df_test[col].fillna(0)



df_test['LotFrontage'] = df_test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



df_test['MSZoning'] = df_test['MSZoning'].fillna('RL')

df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])

df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(df_train['YearBuilt'])
# fill test set na

df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test['YearBuilt'])
#check train na

total0 = df_train.isnull().sum().sort_values(ascending=False)

percent0 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data0 = pd.concat([total0, percent0], axis=1, keys=['Total0', 'Percent0'])

missing_data0.head(20)
#check test na

total1 = df_test.isnull().sum().sort_values(ascending=False)

percent1 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total1', 'Percent1'])

missing_data1.head(20)
df_train['remod'] = np.where(df_train['YearBuilt']==df_train['YearRemodAdd'],0,1)

df_train['age']=df_train['YrSold']-df_train['YearRemodAdd']

df_train['new']=np.where(df_train['YrSold']==df_train['YearBuilt'],0,1) # is new house?

df_train['TotalBath']=df_train['BsmtFullBath']+0.5*df_train['BsmtHalfBath']+df_train['FullBath']+0.5*df_train['HalfBath']

df_train['TotalSqFeet']=df_train['GrLivArea']+df_train['TotalBsmtSF']

df_train['TotalPorchSF']=df_train['OpenPorchSF']+df_train['EnclosedPorch']+df_train['3SsnPorch']+df_train['ScreenPorch']
df_test['remod'] = np.where(df_test['YearBuilt']==df_test['YearRemodAdd'],0,1)

df_test['age']=df_test['YrSold']-df_test['YearRemodAdd']

df_test['new']=np.where(df_test['YrSold']==df_test['YearBuilt'],0,1) # is new house?

df_test['TotalBath']=df_test['BsmtFullBath']+0.5*df_test['BsmtHalfBath']+df_test['FullBath']+0.5*df_test['HalfBath']

df_test['TotalSqFeet']=df_test['GrLivArea']+df_test['TotalBsmtSF']

df_test['TotalPorchSF']=df_test['OpenPorchSF']+df_test['EnclosedPorch']+df_test['3SsnPorch']+df_test['ScreenPorch']
#df_test['TotalBath'] = df_test['TotalBath'].fillna(0.0)

#df_test['TotalSqFeet'] = df_test['TotalSqFeet'].fillna(0.0)
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True);
corr_num = 15 #number of variables for heatmap

cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.show()
#num_df1=df_train[[, 

              #  , , ,'age',',','TotalPorchSF']]
#'OverallQual','TotalSqFeet','GrLivArea','GarageCars','TotalBath','GarageArea','TotalBsmtSF', '1stFlrSF', 'FullBath','YearBuilt',

#'GarageYrBlt','YearRemodAdd','TotRmsAbvGrd'
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
X_train=df_train[['OverallQual', 'GrLivArea', 'TotalSqFeet', 'GarageCars','TotalBath','GarageArea','TotalBsmtSF',

'1stFlrSF','FullBath','YearBuilt','MSSubClass','MSZoning','Neighborhood','LotConfig','YearRemodAdd','TotRmsAbvGrd',

'HeatingQC','KitchenQual','FireplaceQu','OverallQual']]

y_train=df_train['SalePrice']


X_test=df_test[['OverallQual', 'GrLivArea', 'TotalSqFeet', 'GarageCars','TotalBath','GarageArea','TotalBsmtSF',

'1stFlrSF','FullBath','YearBuilt','MSSubClass','MSZoning','Neighborhood','LotConfig','YearRemodAdd','TotRmsAbvGrd',

'HeatingQC','KitchenQual','FireplaceQu','OverallQual']]
#check final feature train set na



total2 = X_train.isnull().sum().sort_values(ascending=False)

percent2 = (X_train.isnull().sum()/X_train.isnull().count()).sort_values(ascending=False)

missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total2', 'Percent2'])

missing_data2.head(20)
#check final feature test set na

total3 = X_test.isnull().sum().sort_values(ascending=False)

percent3 = (X_test.isnull().sum()/X_test.isnull().count()).sort_values(ascending=False)

missing_data3 = pd.concat([total3, percent3], axis=1, keys=['Total3', 'Percent3'])

missing_data3.head(20)
#convert numeric to string

str_vars = ['MSSubClass','YrSold','MoSold']

for var in str_vars:

    df_train[var] = df_train[var].apply(str)

    df_test[var] = df_test[var].apply(str)
#one hot encoding

X_train1 = pd.get_dummies(data=X_train, drop_first=True)

X_train1.head(5)
X_test1 = pd.get_dummies(data=X_test, drop_first=True)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, make_scorer

#lr = LinearRegression()

#lr.fit(X_train1, y_train)



from sklearn.model_selection import cross_val_score, train_test_split

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train1, y_train, scoring = scorer, cv = 10))

    return(rmse)
#lasso regression

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline
lr = LinearRegression()
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

alphas2 = [0.0001, 0.0002, 0.0003, 0.0005, 0.0006, 0.0007]

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                    alphas=alphas2,random_state=42, cv=kfolds))
alphas_alt = [14.6, 14.7,15, 15.1, 15.3, 15.4]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
import lightgbm as lgb

from lightgbm import LGBMRegressor



lgbm = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# store models, scores and prediction values 

models = {'Ridge': ridge,

          'Lasso': lasso, 

          'Linear regression':lr

#          'ElasticNet': elasticnet}

#           'lightgbm': lightgbm,

#           'xgboost': xgboost}

         }

predictions = {}

scores = {}
# model scoring and validation function

def cv_rmse(model, X_train1=X_train1):

    rmse = np.sqrt(-cross_val_score(model, X_train1, y_train, scoring="neg_mean_squared_error",cv=kfolds))

    return (rmse)



# rmsle scoring function

def rmsle(y_train, y_train_pred):

    return np.sqrt(mean_squared_error(y_train, y_train_pred))
for name, model in models.items():

    

    model.fit(X_train1, y_train)

    predictions[name] = np.expm1(model.predict(X_train1))

    

    score = cv_rmse(model, X_train1)

    scores[name] = (score.mean(), score.std())
# get the performance of each model on training data(validation set)

print('---- Score with CV_RMSLE-----')

score = cv_rmse(lr)

print("Linear regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(elastic)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lgbm)

print("lgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


lgbm_fit = lgbm.fit(X_train1, y_train)

y_test_pred = lgbm_fit.predict(X_test1)

#y_test_pred = lgbm_model_fit.predict(X_test1)
#Inverse Logtransforming using np.expm1

y_test_pred_final=np.expm1(y_test_pred)

y_test_pred_final



len(y_test_pred_final)

df_test.shape
my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_test_pred_final})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)