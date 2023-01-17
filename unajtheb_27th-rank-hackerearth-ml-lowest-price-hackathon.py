!pip install sweetviz
# Importing Libraries(Technologies)

from math import sqrt

from pandas_datareader import data 

import matplotlib.pyplot as plt

import scipy

import pandas as pd

import datetime as dt

from datetime import date

import urllib.request,json

import os

import numpy as np

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import sweetviz

from keras.models import Sequential

from keras.layers import Dense,LSTM
df=pd.read_csv('https://github.com/ajtheb/Machine-Learning-Challenges-and-Hackathons/raw/master/Lowest%20price%20hackerearth/Dataset/Train.csv')

dftest=pd.read_csv('https://github.com/ajtheb/Machine-Learning-Challenges-and-Hackathons/raw/master/Lowest%20price%20hackerearth/Dataset/Test.csv')
my_report = sweetviz.analyze([dftest, "Test"])

my_report.show_html('Report.html')
df.head()
dftest.head()
df.describe()
dftest.describe()
df.isnull().sum()
dftest.isnull().sum()
df.boxplot(rot=90,grid=False)
dftest.boxplot(rot=90,grid=False)
import matplotlib.pyplot as plt



x = df['Demand']

y = dftest['Demand']

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(range(len(x)),x, s=10, c='b', marker="s", label='train')

ax1.scatter(range(len(y)),y, s=10, c='r', marker="o", label='test')

plt.legend(loc='upper left');

plt.show()
import matplotlib.pyplot as plt



x = df['High_Cap_Price']

y = dftest['High_Cap_Price']

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(range(len(x)),x, s=10, c='b', marker="s", label='train')

ax1.scatter(range(len(y)),y, s=10, c='r', marker="o", label='test')

plt.legend(loc='upper left');

plt.show()
import matplotlib.pyplot as plt



x = df['Product_Category']

y = dftest['Product_Category']

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(range(len(x)),x, s=10, c='b', marker="s", label='train')

ax1.scatter(range(len(y)),y, s=10, c='r', marker="o", label='test')

plt.legend(loc='upper left');

plt.show()
import matplotlib.pyplot as plt



x = df['Market_Category']

y = dftest['Market_Category']

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(range(len(x)),x, s=10, c='b', marker="s", label='train')

ax1.scatter(range(len(y)),y, s=10, c='r', marker="o", label='test')

plt.legend(loc='upper left');

plt.show()
df['Demand'].sort_values(ascending=False).head(6)
df=df.drop(labels=[1682],axis=0)
plt.scatter(df['Demand'],df['Low_Cap_Price'])
plt.scatter(df['High_Cap_Price'],df['Low_Cap_Price'])
df.groupby('Date')['Low_Cap_Price'].mean().plot(kind='bar')
import seaborn as sns

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(12,12))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn",square=True)
df.groupby(['State_of_Country','Product_Category'])['Low_Cap_Price'].mean().plot(kind='bar')
df['Date']=pd.to_datetime(df['Date'])

dftest['Date']=pd.to_datetime(dftest['Date'])



df['month']=pd.DatetimeIndex(df['Date']).month

df['day']=pd.DatetimeIndex(df['Date']).day

df['year']=pd.DatetimeIndex(df['Date']).year

df['weekday']=pd.DatetimeIndex(df['Date']).weekday

df['day_m_y']=df['day']*df['month']*df['year']

df['quarter'] = df['Date'].dt.quarter

df['weekofyear'] = df['Date'].dt.weekofyear

df['dayofyear'] = df['Date'].dt.dayofyear



dftest['month']=pd.DatetimeIndex(dftest['Date']).month

dftest['day']=pd.DatetimeIndex(dftest['Date']).day

dftest['year']=pd.DatetimeIndex(dftest['Date']).year

dftest['weekday']=pd.DatetimeIndex(dftest['Date']).weekday

dftest['day_m_y']=dftest['day']*dftest['month']*dftest['year']

dftest['quarter'] = dftest['Date'].dt.quarter

dftest['weekofyear'] = df['Date'].dt.weekofyear

dftest['dayofyear'] = df['Date'].dt.dayofyear
df.head()
df.groupby('Grade')['Low_Cap_Price'].mean().sort_values().plot(kind='bar')
pre={

      2:0,

      3:1,

      0:2,

      1:3

}

df['grade_coded']=df['Grade'].map(pre)

dftest['grade_coded']=dftest['Grade'].map(pre)
df['Demand*grade']=((df['Demand']-df['Demand'].min())/(df['Demand'].max()-df['Demand'].min()))*((df['grade_coded']-df['grade_coded'].min())/(df['grade_coded'].max()-df['grade_coded'].min()))

dftest['Demand*grade']=((dftest['Demand']-dftest['Demand'].min())/(dftest['Demand'].max()-dftest['Demand'].min()))*((dftest['grade_coded']-dftest['grade_coded'].min())/(dftest['grade_coded'].max()-dftest['grade_coded'].min()))
ordinal=['State_of_Country','Market_Category','Product_Category']
for f in ordinal:

  s=df.groupby(f)['Low_Cap_Price'].mean().sort_values()

  pre={}

  x=0

  for rownum,(indx,val) in enumerate(s.iteritems()):

    pre[indx]=x

    x+=1

  df[f+'coded']=df[f].map(pre)

  dftest[f+'coded']=dftest[f].map(pre)

  df=df.drop(columns=[f],axis=1)

  dftest=dftest.drop(columns=[f],axis=1)

#ddf['prod_quarter']=df['Product_Categorycoded']*df['quarter']

df['prod_quar']=((df['Product_Categorycoded']-df['Product_Categorycoded'].min())/(df['Product_Categorycoded'].max()-df['Product_Categorycoded'].min()))*((df['quarter']-df['quarter'].min())/(df['quarter'].max()-df['quarter'].min()))

dftest['prod_quar']=((dftest['Product_Categorycoded']-dftest['Product_Categorycoded'].min())/(dftest['Product_Categorycoded'].max()-dftest['Product_Categorycoded'].min()))*((dftest['quarter']-dftest['quarter'].min())/(dftest['quarter'].max()-dftest['quarter'].min()))
df.head()
#df['Market_product']=df['Market_Category']*df['Product_Category']

df['Market_product']=((df['Product_Categorycoded']-df['Product_Categorycoded'].min())/(df['Product_Categorycoded'].max()-df['Product_Categorycoded'].min()))*((df['Market_Categorycoded']-df['Market_Categorycoded'].min())/(df['Market_Categorycoded'].max()-df['Market_Categorycoded'].min()))

dftest['Market_product']=((dftest['Product_Categorycoded']-dftest['Product_Categorycoded'].min())/(dftest['Product_Categorycoded'].max()-dftest['Product_Categorycoded'].min()))*((dftest['Market_Categorycoded']-dftest['Market_Categorycoded'].min())/(dftest['Market_Categorycoded'].max()-dftest['Market_Categorycoded'].min()))
plt.scatter(df['Demand*grade'],df['Low_Cap_Price'])
df['State_Prod']=df['State_of_Countrycoded']-df['Product_Categorycoded']

dftest['State_Prod']=dftest['State_of_Countrycoded']-dftest['Product_Categorycoded']
pearsoncorr = df.corr(method='pearson')

abs(pearsoncorr['Low_Cap_Price']).sort_values(ascending=False)
X=df.drop(columns=['Low_Cap_Price','Item_Id','Date','day','month','weekday','grade_coded','day_m_y','prod_quar','Grade','year'],axis=1)

y=df['Low_Cap_Price'].values
Datatest=dftest.drop(columns=['Item_Id','Date','day','month','weekday','grade_coded','day_m_y','prod_quar','Grade','year'],axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
import xgboost as xgb

from sklearn.metrics import mean_squared_error,r2_score

xgbr=xgb.XGBRegressor()

xgbr.fit(X_train,y_train)

y=xgbr.predict(X_test)

y=abs(y)

from sklearn.metrics import mean_squared_log_error

print('RMSLE:', 100-mean_squared_log_error(y_test, y))

print('RMSE:', np.sqrt(mean_squared_error(y_test, y)))

print('RMSE:', r2_score(y_test, y))
from xgboost import plot_importance

plot_importance(xgbr)
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_test, label=y_test)



param = {'objective': 'regression',

         'num_leaves':20,

         'boosting': 'gbdt',  

         'metric': 'mae',

         'learning_rate': 0.2, 

         'num_iterations': 1000,

         'num_leaves': 80,

         'max_depth': 6,

         'min_data_in_leaf': 11,

         'bagging_fraction': 0.80,

         'bagging_freq': 1,

         'bagging_seed': 142,

         'feature_fraction': 0.80,

         'feature_fraction_seed': 2,

         'early_stopping_round': 200,

         'max_bin': 250

         }



lgbm = lgb.train(params=param, verbose_eval=100, train_set=train_data, valid_sets=[test_data])



y_pred_lgbm = lgbm.predict(X_test)

y_pred_lgbm=abs(y_pred_lgbm)

#print('RMSLE:', sqrt(mean_absolute_error(np.expm1(y_cv), np.expm1(y_pred_lgbm))))

from sklearn.metrics import mean_squared_log_error

print('RMSLE:', 100-mean_squared_log_error(y_test, y_pred_lgbm))

import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(), X.columns), reverse=True)[:50], 

                           columns=['Value','Feature'])

plt.figure(figsize=(12, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
# Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20] ,

 "max_depth"        : [ 3,4,5,6,8,10,13,15],

 "min_child_weight" : [  3,5, 6,9 ,11,14,19],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ,0.5],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7,0.9 ],

 "num_boost_round": [10, 25, 50,80,100,150],

  "n_estimators" :[50,100,200,300,500] 

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

random_search=RandomizedSearchCV(xg_reg,param_distributions=params,n_iter=5,n_jobs=1,cv=4,verbose=3)
from datetime import datetime

# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X_train,y_train)

timer(start_time) # timing ends here for "start_time" variable
random_search.best_params_
random_search.best_estimator_
random_search.best_score_
xgbr=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.7, gamma=0.1, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.05, max_delta_step=0, max_depth=13,

             min_child_weight=19,  monotone_constraints='()',

             n_estimators=500, n_jobs=0, num_boost_round=50,

             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,

             scale_pos_weight=1, subsample=1, tree_method='exact',

             validate_parameters=1, verbosity=None)

xgbr.fit(X_train,y_train)

y=xgbr.predict(X_test)

y=abs(y)

print('RMSLE:', 100-mean_squared_log_error(y_test, y))

print('RMSE:', np.sqrt(mean_squared_error(y_test, y)))

print('r2 score:', r2_score(y_test, y))
from xgboost import plot_importance

plot_importance(xgbr)
from matplotlib.pyplot import figure

figure(figsize=(14,14))

plt.scatter(range(len(y)),y,label='predicted',c='r')

plt.scatter(range(len(y_test)),y_test,label='Actual',c='g')
out=xgbr.predict(Datatest)

out=abs(out)
dataf=pd.DataFrame({

    'Item_Id':dftest['Item_Id'],

    'Low_Cap_Price':out

})
dataf.to_csv('lowest_price_sub.csv',index=False)