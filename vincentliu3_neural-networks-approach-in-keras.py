import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(1126)
train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
weather_df = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')
def TripToDatatime(df):

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['pickup_date'] = df.pickup_datetime.dt.date

    df['pickup_month'] = df.pickup_datetime.dt.month

    df['pickup_day'] = df.pickup_datetime.dt.day

    df['pickup_hour'] = df.pickup_datetime.dt.hour

    df['pickup_minute'] = df.pickup_datetime.dt.minute

    df['day_of_week'] = df.pickup_datetime.dt.weekday

    return df



def WeatherToDatatime(df):

    df['date'] = pd.to_datetime(df['date']).dt.date

    #df['timestamp'] = pd.to_datetime(df['timestamp'])

    #df['weather_date'] = df.timestamp.dt.date

    #df['weather_hour'] = df.timestamp.dt.hour

    return df
train_df = TripToDatatime(train_df)

test_df = TripToDatatime(test_df)

weather_df = WeatherToDatatime(weather_df)



train_df = pd.merge(train_df, weather_df,  how='left', left_on=['pickup_date'], right_on = ['date'])

train_df = train_df.drop(['pickup_datetime', 'dropoff_datetime', 'pickup_date', 'date'], 1)

# train_df = pd.merge(train_df, weather_df,  how='left', left_on=['pickup_date', 'pickup_hour'], right_on = ['weather_date', 'weather_hour'])

# train_df = train_df.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'pickup_date', 'timestamp', 'weather_date', 'weather_hour'], 1)

train_df.head()
train_distance_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])

train_distance_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

test_distance = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])



train_distance = pd.concat((train_distance_1, train_distance_2))

train_df = train_df.merge(train_distance, how='left', on='id')

test_df = test_df.merge(test_distance, how='left', on='id')

train_df.head()
# categorical_features = ['vendor_id','store_and_fwd_flag', 'pickup_hour', 'pickup_date', 'pickup_month', 'day_of_week', 'conditions]

# train_df = train_df.drop(['dailyprecip', 'dailysnow'], 1) # delte columns with unknow meaning



categorical_features = ['vendor_id', 'store_and_fwd_flag']

train_df = pd.get_dummies(train_df, columns=categorical_features)

train_df.head()
unit_features = ['vendor_id_1', 'vendor_id_2', 'store_and_fwd_flag_N', 'store_and_fwd_flag_Y']

for c in unit_features:

    train_df[c] = train_df[c].astype(float)
train_df['precipitation'] = [ 0.01 if c=='T' else float(c) for c in train_df['precipitation']]

train_df['snow fall']= [ 0.01 if c=='T' else float(c) for c in train_df['snow fall']]

train_df['snow depth'] = [ 0.01 if c=='T' else float(c) for c in train_df['snow depth']]
print(train_df.columns)

print(train_df.values[0])

# print(train_df.isnull().sum(axis=0).reset_index())

print(train_df.info())
from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from keras import backend as K
X_train, Y_train = train_df.drop('trip_duration', 1), train_df['trip_duration']

X_train = X_train.values

Y_train = Y_train.values
X_train[0]

type(Y_train[0])
def RMSLE(y_true, y_pred):

    return K.sqrt(K.mean(K.square(K.log(y_pred+1) - K.log(y_true+1)), axis=-1)) 



input_dim = X_train.shape[1]



model = Sequential()

model.add(Dense(1, input_shape=(input_dim,)))



model.compile(loss=RMSLE, optimizer='rmsprop')



history = model.fit(X_train, Y_train,

                    batch_size=64, 

                    epochs=5,

                    verbose=1, 

                    validation_split=0.2)
Y_test = model.predict(X_test)

Y_test
# test_df = pd.read_csv('data/test.csv')

len(test_df)
with open('submission.csv', 'w') as fp:

    fp.write('id,trip_duration\n')

    for x in range(len(test_df)):

        fp.write('{},{}\n'.format(test_df['id'][x], Y_test[x]))
submission_file = pd.read_csv('submission.csv')

submission_file.shape

# shoud have 625134 prediction rows