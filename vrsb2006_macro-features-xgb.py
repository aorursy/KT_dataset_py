import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime



#load files

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

id_test = test.id
# Add month-year

month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

train['month_year_cnt'] = month_year.map(month_year_cnt_map)



month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

test['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

train['week_year_cnt'] = week_year.map(week_year_cnt_map)



week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

test['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

train['month'] = train.timestamp.dt.month

train['dow'] = train.timestamp.dt.dayofweek



test['month'] = test.timestamp.dt.month

test['dow'] = test.timestamp.dt.dayofweek



# Other feature engineering

train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)

train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)



test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)

test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)



train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)

test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)
y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))

        #x_test.drop(c,axis=1,inplace=True)     
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.head()
output.to_csv('Sub_feat.csv', index=False)