# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import numpy as np

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, ShuffleSplit
df = pd.read_csv('../input/train.csv')
def split_datetime(df, column_name):

    df[column_name] = pd.to_datetime(df[column_name])

    df['year_' + column_name] = df[column_name].dt.year

    df['month_' + column_name] = df[column_name].dt.month

    df['day_' + column_name] = df[column_name].dt.day

    df['weekday_' + column_name] = df[column_name].dt.weekday + 1

    df['hour_' + column_name] = df[column_name].dt.hour

    df['minute_' + column_name] = df[column_name].dt.minute

    return df
df.head()
new_df = split_datetime(df, 'pickup_datetime')

new_df.shape
# new_df['store_and_fwd_flag'] = pd.get_dummies(new_df['store_and_fwd_flag'])

# new_df.head()
new_df = new_df[new_df['passenger_count'] != 0]

new_df.shape
new_df = new_df[new_df['trip_duration'] <= 1800]

new_df.shape
new_df = new_df[new_df['trip_duration'] >= 360]

new_df.shape
selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                    'dropoff_latitude', 'month_pickup_datetime', 'hour_pickup_datetime']
X_full = new_df[selected_columns]

y_full = new_df['trip_duration']

X_full.shape, y_full.shape
X_train, X_valid, y_train, y_valid = train_test_split(

            X_full, y_full, test_size=0.33, random_state=42)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#X_train, X_valid, y_train, y_valid = train_test_split(

#            X_train_used, y_train_used, test_size=0.33, random_state=50)

#X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#rf = RandomForestRegressor()
xbr = XGBRegressor(n_jobs=-1)
#params_grid = {

#    'colsample_bytree':[0.4,0.6,0.8],

#    'gamma':[0.03,0.3],

#    'min_child_weight':[1.5,6],

#    'learning_rate':[0.1,0.07],

#    'reg_alpha':[1e-5, 1e-2,  0.75],

#    'reg_lambda':[1e-5, 1e-2, 0.45],

#    'subsample':[0.6,0.95],

#    'max_depth': [3, 5],

    #'min_samples_leaf': [1, 3, 8, 12]

#}
# kf = KFold(n_splits=5, random_state=1) 
# gsc = GridSearchCV(xbr, params_grid, n_jobs=-1, verbose=10, scoring='neg_mean_squared_log_error')
# gsc.fit(X_train, y_train)
# gsc.estimator
# gsc.best_index_
# gsc.best_score_
# gsc.best_params_
# gsc.n_splits_
#cv = ShuffleSplit(n_splits=5, train_size=0.75, random_state=0)
# rf_v2 = RandomForestRegressor(max_depth=15, min_samples_leaf=12)
xbr.fit(X_train, y_train)
xbr.feature_importances_
#losses = -cross_val_score(rf_v2, X_train_used, y_train_used, cv=gsc.best_index_, scoring='neg_mean_squared_log_error')

#losses.mean()
#losses = [np.sqrt(l) for l in losses]

#np.mean(losses)
#rf_v2.fit(X_train_used, y_train_used)
#rf_v2.feature_importances_
y_pred_valid = xbr.predict(X_valid)
#y_pred = rf_v2.predict(X_train_unused)
y_pred_valid.mean()
np.mean(y_valid)
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test = split_datetime(df_test, 'pickup_datetime')
# df_test['store_and_fwd_flag'] = pd.get_dummies(df_test['store_and_fwd_flag'])
X_test = df_test[selected_columns]
y_pred_test = xbr.predict(X_test)
y_pred_test.mean()
submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred_test
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls