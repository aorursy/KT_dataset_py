import os

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

os.listdir('../input/')
def remove_outliers(df):
    df = df.dropna()
    mask = df['fare_amount'].between(0, 500)
    mask &= df['passenger_count'].between(0, 6)

    # to select coordinates: https://www.openstreetmap.org/export
    mask &= df['pickup_longitude'].between(-75, -73)
    mask &= df['dropoff_longitude'].between(-75, -73)
    mask &= df['pickup_latitude'].between(40, 42)
    mask &= df['dropoff_latitude'].between(40, 42)

    return df[mask]

def manhattan(pickup, dropoff):
    pickup_long, pickup_lat = pickup
    dropoff_long, dropoff_lat = dropoff
    return np.abs(dropoff_long - pickup_long) + np.abs(dropoff_lat - pickup_lat)

def extract_distance_features(df):
    df['abs_diff_longitude'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()
    df['abs_diff_latitude'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()

    pickup = (df['pickup_longitude'], df['pickup_latitude'])
    dropoff = (df['dropoff_longitude'], df['dropoff_latitude'])
    df['distance'] = manhattan(pickup, dropoff)

    # Distances to nearby airports, and city center
    # https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    coordinates = {
        'nyc': (-74.0063889, 40.7141667),
        'jfk': (-73.7822222222, 40.6441666667),
        'ewr': (-74.175, 40.69),
        'lgr': (-73.87, 40.77)
    }

    for name, coord in coordinates.items():
        df[f'pickup_distance_to_{name}'] = manhattan(coord, pickup)
        df[f'dropoff_distance_to_{name}'] = manhattan(coord, dropoff)

    return df

def extract_datetime_features(df):
    # Removing unecessary information from the datetime string
    # https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    pickup_datetime = df['pickup_datetime'].str.slice(0, 16)
    pickup_datetime = pd.to_datetime(pickup_datetime, utc=True, format='%Y-%m-%d %H:%M')

    df['year'] = pickup_datetime.dt.year
    df['month'] = pickup_datetime.dt.month
    df['day'] = pickup_datetime.dt.day
    df['dayofweek'] = pickup_datetime.dt.dayofweek
    df['hour'] = pickup_datetime.dt.hour

    return df.drop(columns='pickup_datetime')

def extract_features(df):
    df = extract_distance_features(df)
    df = extract_datetime_features(df)
    return df
# save some space from the defaults float64 and int64
dtypes = {'fare_amount': 'float32',
          'pickup_datetime': 'str',
          'pickup_longitude': 'float32',
          'pickup_latitude': 'float32',
          'dropoff_longitude': 'float32',
          'dropoff_latitude': 'float32',
          'passenger_count': 'uint8'}

val_size = 10_000
input_path = '../input/train.csv'

val_df = pd.read_csv(input_path, usecols=dtypes.keys(), dtype=dtypes, nrows=val_size)
val_df = remove_outliers(val_df)
val_df = extract_features(val_df)

X_val = val_df.drop(columns='fare_amount')
y_val = val_df[['fare_amount']]
dval = xgb.DMatrix(X_val, y_val, feature_names=X_val.columns)

batch_size = 10_000_000
columns = pd.read_csv(input_path, nrows=0).columns
train_df = pd.read_csv(input_path, usecols=dtypes.keys(), dtype=dtypes,
                       names=columns, skiprows=val_size + 1,
                       chunksize=batch_size)
val_df.dtypes
params = {'learning_rate': 0.05,
          'max_depth': 7,
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'subsample': 0.8,
          'gamma': 1,
          'silent': True,
          'verbose_eval': True}

num_rounds = 100
model = None
for batch_df in tqdm(train_df):
    batch_df = remove_outliers(batch_df)
    batch_df = extract_features(batch_df)

    X_train = batch_df.drop(columns='fare_amount')
    y_train = batch_df[['fare_amount']]
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns)

    model = xgb.train(params, dtrain, num_rounds, early_stopping_rounds=5,
                      evals=[(dtrain, 'train'), (dval, 'eval')],
                      xgb_model=model)
xgb.plot_importance(model)
test_df = pd.read_csv('../input/test.csv')
test_df = extract_features(test_df)
test_df.dtypes
X_test = test_df.drop(columns='key')
dtest = xgb.DMatrix(X_test, feature_names=X_test.columns)
y_pred = model.predict(dtest)

submission = pd.DataFrame({'key': test_df['key'], 'fare_amount': y_pred})
submission.to_csv('submission.csv', index = False)

!head submission.csv