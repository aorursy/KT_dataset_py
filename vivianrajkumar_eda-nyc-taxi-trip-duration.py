# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

from haversine import haversine



%matplotlib inline

%config InlineBackend.figure_format = 'retina'
# Import datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("Shape of training set: ", train.shape)

print("Shape of test set: ", test.shape)
print("Columns in training set:\n", train.columns.values)

print('\n')

print("Columns in test set:\n", test.columns.values)
train.head()
train.info()
train.describe()
# Check for null values

train.isnull().sum()
# Convert timestamps from strings to datetime objects

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
# Check values for store_and_fwd_flag

train['store_and_fwd_flag'].unique()
# Convert store_and_fwd_flag values to boolean values

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].apply(lambda y_or_n: y_or_n == 'Y')

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].apply(lambda y_or_n: y_or_n == 'Y')
# Create pickup_hour and dropoff_hour attributes

train['pickup_hour'] = train['pickup_datetime'].dt.hour

test['pickup_hour'] = test['pickup_datetime'].dt.hour

train['dropoff_hour'] = train['dropoff_datetime'].dt.hour



# Create pickup_day attribute (Monday : 0, Sunday : 6)

train['pickup_day'] = train['pickup_datetime'].dt.dayofweek

test['pickup_day'] = test['pickup_datetime'].dt.dayofweek
# Create pickup_location and dropoff_location attributes

train['pickup_location'] = train[['pickup_latitude', 'pickup_longitude']].apply(tuple, axis=1)

train['dropoff_location'] = train[['dropoff_latitude', 'dropoff_longitude']].apply(tuple, axis=1)



test['pickup_location'] = test[['pickup_latitude', 'pickup_longitude']].apply(tuple, axis=1)

test['dropoff_location'] = test[['dropoff_latitude', 'dropoff_longitude']].apply(tuple, axis=1)
# Create great-circle distance between co-ordinates (in km) attribute

train['distance_km'] = train.apply(lambda df: haversine(df['pickup_location'], df['dropoff_location']), axis=1)
train.head()
# Plot all trip durations

plt.scatter(train.index, train['trip_duration'].sort_values(ascending=True))

plt.xlabel('Trip')

plt.ylabel('Trip duration')
# Get quantile

quantile_trip_duration = train['trip_duration'].quantile(0.99)

shortest_trip_higher_than_quantile = train['trip_duration'][train['trip_duration'] > quantile_trip_duration].min()

print("Shortest trip duration higher than the 99.999% quantile: ", shortest_trip_higher_than_quantile, "seconds")
# Filter by quantile

train = train[train['trip_duration'] < quantile_trip_duration]
# Plot all trip durations

trip_durations_sorted = train['trip_duration'].copy().sort_values(ascending=True)

plt.scatter(train.index, trip_durations_sorted)

plt.xlabel('Trip')

plt.ylabel('Trip duration')
# Plot histogram of trip_duration values

plt.figure(figsize=(12, 8))

sns.distplot(train['trip_duration'])

plt.xlabel('Trip duration')
# Plot histogram of log(trip_duration) values

plt.figure(figsize=(12, 8))

sns.distplot(np.log(train['trip_duration'].values))

plt.xlabel('Logarithm of trip duration')
# Number of trips for each hour of the day

plt.figure(figsize=(12, 4))

sns.countplot('pickup_hour', data=train, color='#4C72B0')

plt.xlabel('Pickup hour')

plt.ylabel('Trip count')
# Number of trips for each day of the week

plt.figure(figsize=(8, 4))

sns.countplot('pickup_day', data=train, color='#4C72B0')

plt.xlabel('Pickup day (Monday = 0)')

plt.ylabel('Trip count')