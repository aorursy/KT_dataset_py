# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/train.csv',nrows=1000000,parse_dates=["pickup_datetime"])
train= data.copy(deep=True)

train.head()
train.drop('key', axis=1, inplace=True)  
train.describe()
#remove fare amounts less than 2

train= train[train['fare_amount']>2]
#check for null values

train.isnull().sum()
#remove nulls

train = train.dropna(how='any',axis=0) 
#lets check again

train.isnull().sum()
#lets see the passenger count

train.passenger_count.unique()

train= train[train['passenger_count']<10]
#lets see the passenger count

train.passenger_count.unique()
train.describe()
plt.figure(figsize=(10,5))

sns.distplot(train['fare_amount'])

plt.title('Fare distribution')
train = train.loc[train['pickup_latitude'].between(40, 42)]

train = train.loc[train['pickup_longitude'].between(-75, -72)]

train = train.loc[train['dropoff_latitude'].between(40, 45)]

train = train.loc[train['dropoff_longitude'].between(-75, -72)]
#Initially we had 1 million rows

train.shape
train['latitude_diff']= (train['pickup_latitude']-train['dropoff_latitude']).abs()

train['longitude_diff']= (train['pickup_longitude']- train['dropoff_longitude']).abs()
train.describe()
#lets see how many of the rows have 0 absolute difference of latitude and longitude.

X=train[(train['latitude_diff'] == 0) & (train['longitude_diff'] == 0)]

X.shape
#lets add L2 and L1 distance as features in our data

train['L2']=  ((train['dropoff_latitude']-train['pickup_latitude'])**2 +

(train['dropoff_longitude']-train['pickup_longitude'])**2)**1/2



train['L1']= ((train['dropoff_latitude']-train['pickup_latitude']) +

(train['dropoff_longitude']-train['pickup_longitude'])).abs()
train.head()
# Lets see the correlation between features created

corr_mat = train.corr()

corr_mat.style.background_gradient(cmap='coolwarm')
test_data= pd.read_csv('../input/test.csv',parse_dates=["pickup_datetime"])
test= test_data.copy(deep=True)

test.describe()
test['latitude_diff']= (test['pickup_latitude']-test['dropoff_latitude']).abs()

test['longitude_diff']= (test['pickup_longitude']- test['dropoff_longitude']).abs()



test['L2']=  ((test['dropoff_latitude']-test['pickup_latitude'])**2 +

(test['dropoff_longitude']-test['pickup_longitude'])**2)**1/2



test['L1']= ((test['dropoff_latitude']-test['pickup_latitude']) +

(test['dropoff_longitude']-test['pickup_longitude'])).abs()
test.describe()

#train.head()
train_x = train.drop(['pickup_datetime','fare_amount'],axis=1)

train_y = train['fare_amount'].values



test_x = test.drop(columns=['pickup_datetime','key'])
train_x.head()

#test_x.head()


from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



LR = LinearRegression()



LR.fit(train_x, train_y)



#making predictions

lr_prediction= LR.predict(test_x)





submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = lr_prediction

submission.to_csv('submission_LR.csv', index=False)

submission.head(20)



from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()

RF.fit(train_x, train_y)

RF_predict = RF.predict(test_x)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = RF_predict

submission.to_csv('submission_RF.csv', index=False)

submission.head(20)



parameters = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'num_leaves': 25,

        'learning_rate': 0.02,

        'max_depth': -1,

        'subsample': 0.8,

        'subsample_freq': 1,

        'colsample_bytree': 0.6,

        'reg_aplha': 1,

        'reg_lambda': 0.001,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'verbose':0

    

    }





import lightgbm as lgbm



train_lgbm = lgbm.Dataset(train_x, train_y, silent=True)



lgbm_model= lgbm.train(parameters, train_lgbm, num_boost_round=500)



lgbm_prediction= lgbm_model.predict(test_x)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = lgbm_prediction

submission.to_csv('submission_LGBM.csv', index=False)

submission.head(20)





import xgboost as xgb



xgb_train = xgb.DMatrix(train_x, label=train_y)

xgb_test = xgb.DMatrix(test_x)





params = {'max_depth':7,

          'eta':1,

          'silent':1,

          'objective':'reg:linear',

          'eval_metric':'rmse',

          'learning_rate':0.05

         }



xgb_model= xgb.train(params, xgb_train,50 )



xgb_prediction = xgb_model.predict(xgb_test)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = xgb_prediction

submission.to_csv('submission_XGB.csv', index=False)

submission.head(20)

train_x['year'] =train['pickup_datetime'].dt.year

train_x['month'] = train['pickup_datetime'].dt.month

train_x['day']=train['pickup_datetime'].dt.day

train_x['day_of_week']=train['pickup_datetime'].dt.dayofweek

train_x['hour']=pd.to_datetime(train['pickup_datetime'], format='%H:%M').dt.hour

train_x.head()
# Doing same thing for test data

test_x['year'] =test['pickup_datetime'].dt.year

test_x['month'] = test['pickup_datetime'].dt.month

test_x['day']=test['pickup_datetime'].dt.day

test_x['day_of_week']=test['pickup_datetime'].dt.dayofweek

test_x['hour']=pd.to_datetime(test['pickup_datetime'], format='%H:%M').dt.hour
test_x.head()
# Lets see the correlation between features created

train_x['fare']= train['fare_amount']

corr_mat_new = train_x.corr()

corr_mat_new.style.background_gradient(cmap='coolwarm')
# We don't need this row anymore

train_x= train_x.drop(['fare'],axis=1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



LR = LinearRegression()



LR.fit(train_x, train_y)



#making predictions

lr_prediction= LR.predict(test_x)





submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = lr_prediction

submission.to_csv('submission_LR_new.csv', index=False)

submission.head(20)
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()

RF.fit(train_x, train_y)



RF_predict = RF.predict(test_x)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = RF_predict

submission.to_csv('submission_RF_new.csv', index=False)

submission.head(20)

parameters = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'num_leaves': 25,

        'learning_rate': 0.02,

        'max_depth': -1,

        'subsample': 0.8,

        'subsample_freq': 1,

        'colsample_bytree': 0.6,

        'reg_aplha': 1,

        'reg_lambda': 0.001,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'verbose':0

    

    }





import lightgbm as lgbm



train_lgbm = lgbm.Dataset(train_x, train_y, silent=True)



lgbm_model= lgbm.train(parameters, train_lgbm, num_boost_round=500)



#lgbm prediction

lgbm_prediction= lgbm_model.predict(test_x)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = lgbm_prediction

submission.to_csv('submission_LGBM_new.csv', index=False)

submission.head(20)



import xgboost as xgb



xgb_train = xgb.DMatrix(train_x, label=train_y)

xgb_test = xgb.DMatrix(test_x)





params = {'max_depth':7,

          'eta':1,

          'silent':1,

          'objective':'reg:linear',

          'eval_metric':'rmse',

          'learning_rate':0.05

         }



xgb_model= xgb.train(params, xgb_train,50 )



xgb_prediction = xgb_model.predict(xgb_test)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = xgb_prediction

submission.to_csv('submission_XGB_new.csv', index=False)

submission.head(20)
