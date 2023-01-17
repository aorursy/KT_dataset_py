# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, ShuffleSplit
df = pd.read_csv('../input/train.csv', index_col='id')
def split_datetime(df, column_name):

    df[column_name] = pd.to_datetime(df[column_name])

    df['year_' + column_name] = df[column_name].dt.year

    df['month_' + column_name] = df[column_name].dt.month

    df['day_' + column_name] = df[column_name].dt.day

    df['weekday_' + column_name] = df[column_name].dt.weekday

    df['hour_' + column_name] = df[column_name].dt.hour

    df['minute_' + column_name] = df[column_name].dt.minute

    return df
df.head()
new_df = split_datetime(df, 'pickup_datetime')

new_df.shape
new_df = new_df[new_df['passenger_count'] >= 1]

new_df.shape
new_df = new_df[new_df['trip_duration'] <= 7200]

new_df.shape
new_df = new_df[new_df['trip_duration'] >= 300]

new_df.shape
selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                   'dropoff_latitude', 'day_pickup_datetime',

                   'hour_pickup_datetime', 'minute_pickup_datetime']
X_full = new_df[selected_columns]

y_full = new_df['trip_duration']

X_full.shape, y_full.shape
X_train_used, X_train_unused, y_train_used, y_train_unused = train_test_split(

            X_full, y_full, test_size=0.60, random_state=50)

X_train_used.shape, X_train_unused.shape, y_train_used.shape, y_train_unused.shape
X_train, X_valid, y_train, y_valid = train_test_split(

            X_train_used, y_train_used, test_size=0.33, random_state=50)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
rf = RandomForestRegressor()
params_grid = {

    'max_depth': [1, 3, 5, 10, 15],

    'min_samples_leaf': [1, 3, 8, 12]

}
# kf = KFold(n_splits=5, random_state=1)
# gsc = GridSearchCV(rf, params_grid, n_jobs=-1, cv=kf, verbose=3, scoring='neg_mean_squared_log_error')#
# gsc.fit(X_train, y_train)
# gsc.best_estimator_
# gsc.best_index_
cv = ShuffleSplit(1, test_size=0.01, train_size=0.5, random_state=0)
losses = -cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

losses.mean()
losses = [np.sqrt(l) for l in losses]

np.mean(losses)
rf.fit(X_train, y_train)
rf.feature_importances_
y_pred = rf.predict(X_valid)
y_pred.mean()
np.mean(y_valid)
df_test = pd.read_csv('../input/test.csv', index_col='id')
df_test.head()
df_test = split_datetime(df_test, 'pickup_datetime')
X_test = df_test[selected_columns]
y_pred_test = rf.predict(X_test)
y_pred_test.mean()
submission = pd.read_csv('../input/sample_submission.csv', index_col='id') 

submission.head()
submission['trip_duration'] = y_pred_test
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls