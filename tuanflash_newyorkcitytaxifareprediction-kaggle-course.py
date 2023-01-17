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
import seaborn as sns
import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv', nrows=2000000, parse_dates=['pickup_datetime'])
print(train_df.info())
print(train_df.shape)
train_df.head(5)
test_df = pd.read_csv('../input/test.csv')
print(test_df.info())
print(test_df.shape)
test_df.head(5)
print('check nan')
print('pickup_longitude', train_df['pickup_longitude'].isnull().values.sum())
print('pickup_latitude',  train_df['pickup_latitude'].isnull().values.sum())
print('dropoff_longitude', train_df['dropoff_longitude'].isnull().values.sum())
print('dropoff_latitude', train_df['dropoff_latitude'].isnull().values.sum())

train_df['dropoff_longitude'].fillna(train_df['dropoff_longitude'].mean(), inplace=True)
train_df['dropoff_latitude'].fillna(train_df['dropoff_latitude'].mean(), inplace=True)
# visualize and check outlier

# check outlier fare amount
fig, ax = plt.subplots()
train_df.fare_amount.hist(ax=ax, bins=120, figsize=(14,3))
ax.set_yscale('log')
plt.xlabel('fare $USD')
plt.title('Histogram')
# check outlier long, lat
#Plot variables using only 1000 rows for efficiency
train_df.iloc[:1000].plot.scatter('pickup_longitude', 'pickup_latitude')
train_df.iloc[:1000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
train_df.describe()
#Clean dataset
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

train_df = clean_df(train_df)
print(len(train_df))
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    
    return dataset

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))   # 2*R*asin..

def transform(data):
    # Distances to nearby airports, 
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)

    data['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    return data
train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 
                                   train_df['dropoff_latitude'] , train_df['dropoff_longitude'])

train_df = add_datetime_info(train_df)
train_df = transform(train_df)
train_df.head()
test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])
test_df = add_datetime_info(test_df)
test_df = transform(test_df)
test_df.head()
df_train = train_df.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
df_test = test_df.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('fare_amount', axis=1),
                                                    df_train['fare_amount'], test_size=0.2, random_state = 42)
# train xgboost model
import xgboost as xgb
params = {'max_depth': 12, 'gamma': 0.001, 'min_child_weight': 8.0, 'max_delta_step': 10.0, 'subsample': 0.4, 'colsample_bytree': 1.0}
def XGBmodel(X_train,X_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(X_train,label=y_train)
    matrix_test = xgb.DMatrix(X_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=250, early_stopping_rounds=10 ,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(X_train,X_test,y_train,y_test,params)
import matplotlib.pyplot as plt
fscores = pd.DataFrame({'X': list(model.get_fscore().keys()), 'Y': list(model.get_fscore().values())})
fscores.sort_values(by='Y').plot.bar(x='X')
# train lightgbm model
import lightgbm as lgb
y_test = model.predict(xgb.DMatrix(df_test),ntree_limit = model.best_ntree_limit)
print(y_test)
print(y_test.shape)
test_key = test_df['key']
#Create submission file
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": y_test.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission