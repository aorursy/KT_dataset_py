import numpy as np # linear algebra
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_datapath = '../input/demand-forecasting-kernels-only/train.csv'
test_datapath = '../input/demand-forecasting-kernels-only/test.csv'
submission_datapath = '../input/demand-forecasting-kernels-only/sample_submission.csv'
df_train = pd.read_csv(train_datapath)
df_test = pd.read_csv(test_datapath)
df_submission = pd.read_csv(submission_datapath)
df_train.head()
df_test.head()
df_submission.head()
def convert_dates(x):
    x['date']=pd.to_datetime(x['date']) #converting date column to datetime format
    x['month']=x['date'].dt.month #creating a new column 'month' from 'date' using dt.month
    x['year']=x['date'].dt.year #same - for year
    x['dayofweek']=x['date'].dt.dayofweek #same - for day
    x.pop('date') #delete 'date' column
    return x
df_train = convert_dates(df_train)
df_train.head()
df_test = convert_dates(df_test)
df_test.head()
def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean') #daily_avg column based on sales per day
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean') #monthly_avg column based on sales per month
    return x
df_train = add_avg(df_train)
df_train.head()
daily_avg = df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index() #finding daily_avg value to use in x_pred
monthly_avg = df_train.groupby(['item','store','month'])['sales'].mean().reset_index() #finding monthly_avg value to use in x_pred
def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x
df_test = merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test = merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')
df_test.sample(10)
x_train,x_test,y_train,y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'),random_state=123,test_size=0.2) #splitting train dataset to test/train
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'} #reg:linear cuz target value is a regression, mae for mean absolute error, can be rmse as well. More info - see documentation
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],) #early_stopping_rounds = 20 : stop if 20 consequent rounds without decrease of error
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
x_test_pred = model.predict(xgb.DMatrix(x_test))
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
mean_squared_error(y_true=y_test,
                   y_pred=x_test_pred)
root_mean_sqaure_error_RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=x_test_pred))
root_mean_sqaure_error_RMSE
mean_absolute_error(y_true=y_test,
                   y_pred=x_test_pred)
submission = pd.DataFrame(df_test.pop('id'))
submission.head()
y_pred = model.predict(xgb.DMatrix(df_test), ntree_limit = model.best_ntree_limit) #best_ntree_limit derives from best iteration in the model which is 87. For that, need to enable early stopping in the model.
submission['sales']= y_pred
submission.to_csv('submission.csv',index=False)
submission.head()