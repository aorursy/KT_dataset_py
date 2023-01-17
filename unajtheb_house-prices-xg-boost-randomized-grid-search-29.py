# Importing Libraries(Technologies)

from math import sqrt

from pandas_datareader import data 

import matplotlib.pyplot as plt

import scipy

import pandas as pd

import datetime as dt

import urllib.request,json

import os

import numpy as np

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

#import sweetviz

from keras.models import Sequential

from keras.layers import Dense,LSTM
!pip install sweetviz
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

dftest=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.shape,dftest.shape
df.head()
dftest.head()
df.describe()
df.dtypes
df.info()
my_report = sweetviz.analyze([df, "Train"],target_feat='SalePrice')

my_report.show_html('Report.html')
df.isnull().sum().sort_values(ascending=False)
#Total null values

df.isnull().values.sum()
cat=df.select_dtypes(include='object').columns

print(cat)
num=df.select_dtypes(include=['int64','float64']).columns

print(num)
df['YrSold'].median()
#filling na values with mode

for x in cat: 

  df[x]=df[x].fillna(df[x].value_counts().index[0])

  dftest[x]=dftest[x].fillna(dftest[x].value_counts().index[0])

#filling na values with median

for x in num:

  df[x]=df[x].fillna(df[x].median())

  if x!='SalePrice':

    dftest[x]=dftest[x].fillna(dftest[x].median())

import matplotlib.pyplot as plt

fig,a =  plt.subplots(7,7,figsize=(16,16))

k=0

fig.tight_layout(pad=3.0)



for x in range(0,7):

  for y in range(0,7):

    a[x][y].bar(df[cat[k]].unique(),df.groupby(cat[k])['SalePrice'].mean())

    a[x][y].set_title(cat[k])

    a[x][y].tick_params(axis='x', labelrotation=90 )

    k+=1

    if(k>42):

      break
ordinal=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']
for f in ordinal:

  s=df.groupby(f)['SalePrice'].mean().sort_values()

  pre={}

  x=0

  for rownum,(indx,val) in enumerate(s.iteritems()):

    pre[indx]=x

    x+=1

  df[f+'coded']=df[f].map(pre)

  dftest[f+'coded']=dftest[f].map(pre)

  df=df.drop(columns=[f],axis=1)

  dftest=dftest.drop(columns=[f],axis=1)

df.head()
pearsoncorr = df.corr(method='pearson')

abs(pearsoncorr['SalePrice'])
df_num=df[num]
X=df.drop(columns=['SalePrice'],axis=1)

y=df['SalePrice']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
import xgboost as xgb

from sklearn.metrics import mean_squared_error
# Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3,5,6,8,10,13,15],

 "min_child_weight" : [ 1, 3, 5, 7,9 ,11,14,19],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ,0.5],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7,0.9 ],

 "num_boost_round": [10, 25, 50,80],

  "n_estimators" :[50,100,200,500,1000,2000] 

}
xg_reg = xgb.XGBRegressor()
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

random_search=RandomizedSearchCV(xg_reg,param_distributions=params,n_iter=5,n_jobs=1,cv=2,verbose=3)
from datetime import datetime

# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X_train,y_train)

timer(start_time) # timing ends here for "start_time" variable
random_search.best_params_
random_search.best_estimator_
random_search.best_score_
xgbr=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.5, gamma=0.4,

             importance_type='gain', learning_rate=0.05, max_delta_step=0,

             max_depth=8, min_child_weight=5, missing=None, n_estimators=2000,

             n_jobs=1, nthread=None, num_boost_round=10, objective='reg:linear',

             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

             seed=None, silent=None, subsample=1, verbosity=1)
xgbr.fit(X_train,y_train)

y=xgbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y))

print("RMSE: %f" % (rmse))
from sklearn.metrics import r2_score

r2_score(y_test,y)
pd.DataFrame({

    'actual':y_test,

    'predicted':y

})
y=xgbr.predict(dftest)
dataf=pd.DataFrame({

    'Id':dftest['Id'],

    'SalePrice':y

})
dataf.head()
dataf.to_csv('house_prices_submission.csv',index=False)