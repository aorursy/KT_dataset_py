# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# display all outputs in jupyter notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Any results you write to the current directory are saved as output.
# we take a subsample of the data to avoid too much computation
train_data = pd.read_csv(r'../input/train.csv', nrows=50000)
train_data.head()
train_data.dtypes
bins = np.linspace(0,max(train_data['fare_amount']),100)
fare_hist = plt.hist(train_data['fare_amount'], bins=bins)
plt.show()
train_data['fare_amount'].describe()
# we notice that the min price is negative : -5

# how many negative fares are negatives ?
(train_data['fare_amount']<0).sum()

# let's see them in details
negative_fares = train_data[train_data['fare_amount']<0]
negative_fares

# let's see these rides on a map
colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink']
m = folium.Map(location=[40.790112, -74.000031], tiles='Stamen Toner')
for i in range(len(negative_fares)):
    pickup_lat = negative_fares.iloc[i]['pickup_latitude']
    pickup_long = negative_fares.iloc[i]['pickup_longitude']
    a = folium.Circle(
        radius=100,
        location=[pickup_lat,pickup_long],
        popup='The Waterfront',
        color=colors[i],
        fill=False,
    ).add_to(m)
    dropoff_lat = negative_fares.iloc[i]['dropoff_latitude']
    dropoff_long = negative_fares.iloc[i]['dropoff_longitude']
    b = folium.Circle(
        radius=100,
        location=[dropoff_lat, dropoff_long],
        popup='The Waterfront',
        color=colors[i],
        fill=False
    ).add_to(m)
m

# we see that the taxi rides with negative fare amounts are either
# - small rides
# - absurd (purple dot at 0 0)
# we decide to get rid of these rows because we consider taxi fares 
# cannot be negatives
train_data = train_data[train_data['fare_amount'] >= 0]
columns = train_data.columns
num_missing = train_data.isnull().sum()
num_zero = pd.Series()
for col in columns:
    num_zero.at[col] = (train_data[col]==0).sum()
missing_value_df = pd.DataFrame({
                                 'num_missing': num_missing,
                                 'num_zero':  num_zero
                                })
missing_value_df

train_data = train_data.drop(train_data[train_data.pickup_longitude==0].index)
train_data = train_data.drop(train_data[train_data.pickup_latitude==0].index)
train_data = train_data.drop(train_data[train_data.dropoff_longitude==0].index)
train_data = train_data.drop(train_data[train_data.dropoff_latitude==0].index)
train_data = train_data.reset_index(drop=True)
import folium

mean_pickup_latitude = np.mean(train_data['pickup_latitude'])
mean_pickup_longitude = np.mean(train_data['pickup_longitude'])

m = folium.Map(location=[mean_pickup_latitude, mean_pickup_longitude], tiles='Stamen Toner')
# we only display 1000
for i in range(len(train_data.sample(frac=0.02))):
    pickup_lat = train_data.at[i,'pickup_latitude']
    pickup_long = train_data.at[i,'pickup_longitude']
    a = folium.Circle(
        radius=10,
        location=[pickup_lat,pickup_long],
        popup='The Waterfront',
        color='crimson',
        fill=False,
    ).add_to(m)
m
# we notice some dots in the water 
from folium import plugins
from folium.plugins import HeatMap

heat_data = []
for i in range (len(train_data.sample(frac=0.02))):
    heat_data.append([train_data.at[i,'pickup_latitude'], train_data.at[i,'pickup_longitude']])

m = folium.Map(location=[mean_pickup_latitude, mean_pickup_longitude])
HeatMap(heat_data).add_to(m)
m
for i in range(len(train_data)):
    train_data.at[i,'pickup_datetime'] = train_data.at[i,'pickup_datetime'].split(' UTC')[0]
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train_data.at[0,'pickup_datetime'].year
# evolution of the traffic over the years 
from folium.plugins import HeatMapWithTime
train_data.dtypes
max(train_data['pickup_datetime'].dt.year)
min(train_data['pickup_datetime'].dt.year)
heat_data_year = [[[row['pickup_latitude'], row['pickup_longitude']] \
                  for index, row in train_data[train_data['pickup_datetime'].dt.year == i].iterrows()] \
                 for i in range(2009, 2016)]
m = folium.Map(location=[mean_pickup_latitude, mean_pickup_longitude])
HeatMapWithTime(heat_data_year).add_to(m)
m

# taxi work areas seem to have shrunk over the years
# two hubs external of the city can be noticed : the LaGuardia and JFK airports
# evolution of the traffic over the months for 2015
heat_data_month = [[[row['pickup_latitude'], row['pickup_longitude']] \
                    for index, row in train_data[(train_data['pickup_datetime'].dt.year == 2013) & (train_data['pickup_datetime'].dt.month == i)].iterrows()] \
                   for i in range(1,13)]
m = folium.Map(location=[mean_pickup_latitude, mean_pickup_longitude])
HeatMapWithTime(heat_data_month, auto_play=True).add_to(m)
m
# evolution of the traffic on a day 
heat_map_hour = [[[row['pickup_latitude'], row['pickup_longitude']] \
                 for index, row in train_data[(train_data['pickup_datetime'].dt.year == 2015) & \
                                             (train_data['pickup_datetime'].dt.month == 5) & \
                  (train_data['pickup_datetime'].dt.day == 19) & (train_data['pickup_datetime'].dt.hour == i)].iterrows()] for i in range(0,24)]
HeatMapWithTime(heat_map_hour, auto_play=True).add_to(m)
m
# Let's compute the distance between the pickup and dropoff places 
# First we need to compute the Haversine distance as we use the latitude and longitude
# We use the formula given : https://www.movable-type.co.uk/scripts/latlong.html

def haversine_distance(lat1, lat2, long1, long2):
    # R is the Earth radius
    R = 6371
    delta_lat = (lat1 - lat2)*np.pi/180.
    delta_long = (long1 - long2)*np.pi/180
    a = np.sin(delta_lat / 2.)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_long / 2.)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

train_data['h_dist'] = haversine_distance(train_data['pickup_latitude'], train_data['dropoff_latitude'], \
                                          train_data['pickup_longitude'], train_data['dropoff_longitude'])

train_data['h_dist'].describe()
train_data['h_dist'].median()

# We can notice extemely long distances (max at near 20000 kms) which is very suspicious
# especially when considering the price
# We drop too big values
train_data = train_data.drop((train_data[train_data['h_dist']>200]).index)


fig, ax = plt.subplots(1,2, figsize=[10,3])
bins = np.linspace(min(train_data['h_dist']), max(train_data['h_dist']), 100)
bins_focus = np.linspace(min(train_data['h_dist']), 0.1*max(train_data['h_dist']), 100)
a = ax[0].hist(train_data['h_dist'], bins)
b = ax[1].hist(train_data['h_dist'], bins_focus)

train_data[train_data['h_dist']==max(train_data['h_dist'])]

# We notice that there are a lot of distances at zero
# which correspond to mistakes
# We drop these values 
train_data = train_data.drop(train_data[train_data['h_dist']==0].index)

# We will check the fare according to the distance
# Now we plot the fare according to the distance 

fig, ax = plt.subplots(2,2, figsize=(20,20))
a = ax[0,0].set_xlabel('distance')
a = ax[0,0].set_ylabel('prix')
a = ax[0,0].scatter(train_data['h_dist'],train_data['fare_amount'])


# Zoom In
idx_zoom = (train_data['h_dist']<train_data['h_dist'].quantile(0.75)) & \
(train_data['fare_amount'] < 100)
a = ax[0,1].set_title('Zoom in')
a = ax[0,1].set_xlabel('distance')
a = ax[0,1].set_ylabel('prix')
a = ax[0,1].scatter(train_data['h_dist'][idx_zoom],train_data['fare_amount'][idx_zoom],\
                  c = np.clip(train_data['fare_amount'][idx_zoom], 0, 100), cmap='viridis')


# distance between 0 and 50 kms

a = ax[1,0].set_xlabel('distance')
a = ax[1,0].set_ylabel('prix')
a = ax[1,0].scatter(train_data['h_dist'][train_data['h_dist']<50],train_data['fare_amount'][train_data['h_dist']<50])


idx_zoom2 = (train_data['h_dist']<30)
a = ax[1,1].set_title('Zoom in')
a = ax[1,1].set_xlabel('distance')
a = ax[1,1].set_ylabel('prix')
a = ax[1,1].scatter(train_data['h_dist'][idx_zoom2],train_data['fare_amount'][idx_zoom2],\
                  c = np.clip(train_data['fare_amount'][idx_zoom2], 0, 100), cmap='viridis')

# We assume that a linear regression could be a good estimate
# though there are outliers with big prices for short distances
# and there is the same price for different distances around 50$
# We should also test the Manhattan distance 
# https://stackoverflow.com/questions/32923363/manhattan-distance-for-two-geolocations
# Indeed Manhattan distance computes a distance based on a grid between two points
# and not just the direct distance like with the Euclidean distance

def manhattan_distance(lat1, lat2, long1, long2):
    # To compute the Manhattan distance we take the latitude distance (considering the longitude distance zero)
    # and the same for the longitude distance
    lat_dist = haversine_distance(lat1, lat2, 0, 0)
    long_dist = haversine_distance(0, 0, long1, long2)
    return lat_dist + long_dist

train_data['m_dist'] = manhattan_distance(train_data['pickup_latitude'], train_data['dropoff_latitude'], \
                                          train_data['pickup_longitude'], train_data['dropoff_longitude'])
train_data['m_dist'].describe()

# We notice that the max distance is very high

fig, ax = plt.subplots(1,2,figsize=[10,3])
bins = np.linspace(min(train_data['m_dist']), max(train_data['m_dist']),100)
bins_zoom =  np.linspace(min(train_data['m_dist']), 0.05*max(train_data['m_dist']),100)
a = ax[0].hist(train_data['m_dist'], bins)
b = ax[1].hist(train_data['m_dist'], bins_zoom)
    
# Now we plot the fare according to the distance 

def plot_fare_distance(dist):
 
    fig, ax = plt.subplots(2,2, figsize=(20,20))
    a = ax[0,0].set_xlabel('distance')
    a = ax[0,0].set_ylabel('prix')
    a = ax[0,0].scatter(train_data[dist],train_data['fare_amount'])


    # Zoom In
    idx_zoom = (train_data[dist]<train_data[dist].quantile(0.75)) & \
    (train_data['fare_amount'] < 100)
    a = ax[0,1].set_title('Zoom in')
    a = ax[0,1].set_xlabel('distance')
    a = ax[0,1].set_ylabel('prix')
    a = ax[0,1].scatter(train_data[dist][idx_zoom],train_data['fare_amount'][idx_zoom],\
                      c = np.clip(train_data['fare_amount'][idx_zoom], 0, 100), cmap='viridis')


    # distance between 0 and 50 kms

    a = ax[1,0].set_xlabel(dist)
    a = ax[1,0].set_ylabel('prix')
    a = ax[1,0].scatter(train_data[dist][train_data[dist]<50],train_data['fare_amount'][train_data[dist]<50])


    idx_zoom2 = (train_data[dist]<30)
    a = ax[1,1].set_title('Zoom in')
    a = ax[1,1].set_xlabel('distance')
    a = ax[1,1].set_ylabel('prix')
    a = ax[1,1].scatter(train_data[dist][idx_zoom2],train_data['fare_amount'][idx_zoom2],\
                      c = np.clip(train_data['fare_amount'][idx_zoom2], 0, 100), cmap='viridis')

    # We assume that a linear regression could be a good estimate
    # though there are outliers with big prices for short distances
    # and there is the same price for different distances around 50$
    
plot_fare_distance('h_dist')
plot_fare_distance('m_dist')
train_df = pd.read_csv(r'../input/train.csv', nrows=5000000)
test_df = pd.read_csv(r'../input/test.csv')

train_df.head()
test_df.head()
train_df = train_df[train_df['fare_amount']>=0]

train_df = train_df.drop(train_df[train_df.pickup_longitude==0].index)
train_df = train_df.drop(train_df[train_df.pickup_latitude==0].index)
train_df = train_df.drop(train_df[train_df.dropoff_longitude==0].index)
train_df = train_df.drop(train_df[train_df.dropoff_latitude==0].index)
train_df = train_df.reset_index(drop=True)
train_df['m_dist'] = manhattan_distance(train_df['pickup_latitude'], train_df['dropoff_latitude'], \
                                        train_df['pickup_longitude'], train_df['dropoff_longitude'])
test_df['m_dist'] = manhattan_distance(test_df['pickup_latitude'], test_df['dropoff_latitude'], \
                                        test_df['pickup_longitude'], test_df['dropoff_longitude'])

for i in range(len(train_df)):
    train_df.at[i,'pickup_datetime'] = train_df.at[i,'pickup_datetime'].split(' UTC')[0]
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['day'] = train_df['pickup_datetime'].dt.day
train_df['hour'] = train_df['pickup_datetime'].dt.hour

for i in range(len(test_df)):
    test_df.at[i,'pickup_datetime'] = test_df.at[i,'pickup_datetime'].split(' UTC')[0]
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['hour'] = test_df['pickup_datetime'].dt.hour

train_df.dropna(inplace=True)
features = ['m_dist']

X = train_df[features]
Y = train_df['fare_amount']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

np.any(np.isnan(X_train))
np.all(np.isfinite(X_train))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin = LinearRegression()
lin.fit(X_train, Y_train)
Y_test_pred = lin.predict(X_test)

print(Y_test_pred)

print('RMSE of the baseline is :')
np.sqrt(mean_squared_error(Y_test, Y_test_pred))


features = ['m_dist', 'passenger_count', 'month', 'day', 'hour']
#features = ['m_dist', 'day']


X = train_df[features]
Y = train_df['fare_amount']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(X_train, Y_train)
print(clf.feature_importances_)
Y_test_pred = clf.predict(X_test)
print(Y_test_pred)
print('RMSE of the the Random Forest Regressor is :')
np.sqrt(mean_squared_error(Y_test, Y_test_pred))
x_test = test_df[features]
y_test_pred = clf.predict(x_test)
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_test_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
submission
