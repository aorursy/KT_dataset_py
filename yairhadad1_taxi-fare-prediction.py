import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split , cross_val_score

from sklearn.linear_model import LinearRegression , SGDRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.cluster import KMeans

from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import make_scorer

from math import radians, cos, sin, atan2, sqrt

import lime

import lime.lime_tabular

import folium

import tensorflow as tf

import xgboost as xgb

import geopy.distance



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



sns.set()
%%time

# Reading File

train_path  = '../input/train.csv'



# Set columns to most suitable type to optimize for memory usage

traintypes = {'fare_amount': 'float32',

              'pickup_datetime': 'str', 

              'pickup_longitude': 'float32',

              'pickup_latitude': 'float32',

              'dropoff_longitude': 'float32',

              'dropoff_latitude': 'float32',

              'passenger_count': 'uint8'}



cols = list(traintypes.keys())



df_train = pd.read_csv(train_path, usecols=cols, dtype=traintypes,nrows=1000000)

df_test = pd.read_csv('../input/test.csv')
df_train.head()
print("Data count:",len(df_train))



print("fare_amount less them zero:",(df_train.fare_amount <= 0).sum())

print("fare amount can't be less then zero so we will remove it\n")

df_train = df_train[df_train.fare_amount > 0]



print("passenger_count esqual zero:", (df_train.passenger_count <= 0).sum())

print("passenger_count bigger then 8:", (df_train.passenger_count > 8).sum())

print("passenger count can't be zero so we will remove it\n")

df_train = df_train.query("passenger_count > 0 & passenger_count <= 8")



print("Latitudes less them -90 or bigger then 90:",(df_train.pickup_latitude < -90).sum(),",",

                                                  (df_train.dropoff_latitude < -90).sum(),",",

                                                  (df_train.pickup_latitude > 90).sum(),",",

                                                  (df_train.dropoff_latitude > 90).sum())



print("Longitudes less them -180 or bigger then 180:",(df_train.pickup_longitude < -180).sum(),",",

                                                  (df_train.dropoff_longitude < -180).sum(),",",

                                                  (df_train.pickup_longitude > 180).sum(),",",

                                                  (df_train.dropoff_longitude > 180).sum())

print("\nby Google maps: \nLatitudes range from -90 to 90\nLongitudes range from -180 to 180\n we will remove them")

df_train['pickup_latitude'] = df_train['pickup_latitude'].astype(float)

df_train['pickup_longitude'] = df_train['pickup_longitude'].astype(float)

df_train['dropoff_latitude'] = df_train['dropoff_latitude'].astype(float)

df_train['dropoff_longitude'] = df_train['dropoff_longitude'].astype(float)

df_train = df_train.query("pickup_latitude <= 90 & pickup_latitude >= -90")

df_train = df_train.query("dropoff_latitude <= 90 & dropoff_latitude >= -90")

df_train = df_train.query("pickup_longitude <= 180 & pickup_longitude >= -180")

df_train = df_train.query("dropoff_longitude <= 180 & dropoff_longitude >= -180")



zero_lan_log_df = df_train.loc[((df_train['pickup_latitude']==0) & (df_train['pickup_longitude']==0)) & ((df_train['dropoff_latitude']==0) & (df_train['dropoff_longitude']==0))]

print("\nLongitudes and Latitudes esqual zero:",len(zero_lan_log_df))

print("Longitudes and Latitudes is zero - This is in the middle of the sea - probably default number - remove them")

df_train = df_train.query("pickup_latitude != 0 | pickup_longitude != 0 | dropoff_latitude != 0 | dropoff_longitude != 0")



print("\ndropoff and pickup in the same location:",((df_train['dropoff_longitude'] == df_train['pickup_longitude']) & (df_train['dropoff_latitude'] == df_train['pickup_latitude'])).sum()," remove them")

df_train = df_train.query("dropoff_longitude != pickup_longitude | dropoff_latitude != pickup_latitude")



print("Data count:",len(df_train))
# extricate day, hour, month, year from pickup_datetime

df_train['pickup_day'] = pd.DatetimeIndex(df_train['pickup_datetime']).day

df_train['pickup_hour'] = pd.DatetimeIndex(df_train['pickup_datetime']).hour

df_train['pickup_week'] = pd.DatetimeIndex(df_train['pickup_datetime']).week

df_train['pickup_month'] = pd.DatetimeIndex(df_train['pickup_datetime']).month

df_train['pickup_year'] = pd.DatetimeIndex(df_train['pickup_datetime']).year



# convert day, hour and month to X and Y on the Unit Circle 

df_train['x_pickup_month']=np.sin(2.*np.pi*df_train.pickup_month/12.)

df_train['y_pickup_month']=np.cos(2.*np.pi*df_train.pickup_month/12.)



dayfloat = df_train.pickup_day+df_train.pickup_hour/24.0

df_train['x_pickup_day']=np.sin(2.*np.pi*dayfloat/31.)

df_train['y_pickup_day']=np.cos(2.*np.pi*dayfloat/31.)



df_train['x_pickup_week']=np.sin(2.*np.pi*df_train.pickup_week/52.)

df_train['y_pickup_week']=np.cos(2.*np.pi*df_train.pickup_week/52.)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,5))



ax = sns.scatterplot(x="x_pickup_month", y="y_pickup_month", hue="fare_amount",alpha = 0.2,palette = 'coolwarm',ax = axes[0], data=df_train)

ax.set_title("Pickup Month")



ax = sns.scatterplot(x="x_pickup_week", y="y_pickup_week", hue="fare_amount",alpha = 0.2,palette = 'coolwarm',ax = axes[1], data=df_train)

ax.set_title("Pickup Week")



ax = sns.scatterplot(x="x_pickup_day", y="y_pickup_day", hue="fare_amount",alpha = 0.2,palette = 'coolwarm',ax = axes[2], data=df_train)

ax.set_title("Pickup Day + Hour")



df_train.groupby('pickup_year').sum()['fare_amount'].plot.bar(title= "Pickup Year count",ax = axes[3])

plt.show()
df_train.groupby('passenger_count').mean()['fare_amount'].plot.bar()



print("Max passenger:",df_train['passenger_count'].max())

print("Min passenger:",df_train['passenger_count'].min())
plt.figure(figsize=(10,10))

ax = sns.scatterplot(x="pickup_longitude", y="pickup_latitude", hue="fare_amount" ,size="fare_amount",data=df_train)

ax = sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude",hue="fare_amount", size="fare_amount",data=df_train)

ax.set_title("Full pickup and dropoff map")

plt.xticks(np.arange(min(df_train['pickup_longitude']), max(df_train['dropoff_longitude'])+1, 20.0))

plt.yticks(np.arange(min(df_train['dropoff_latitude']), max(df_train['dropoff_latitude'])+1, 20.0))

plt.show()
k_mean_X = df_train[["dropoff_longitude","dropoff_latitude"]].copy()

k_mean_y = df_train['fare_amount'].copy()



Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(k_mean_X)

    Sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()



cls_k_means = KMeans(n_clusters=3)

cls_k_means.fit(k_mean_X)

with np.printoptions(suppress=True):

    print(cls_k_means.cluster_centers_)
plt.figure(figsize=(10,10))

ax = sns.scatterplot(x="pickup_longitude", y="pickup_latitude", hue="fare_amount" ,size="fare_amount",data=df_train)

ax = sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude",hue="fare_amount", size="fare_amount",data=df_train)

ax.set_title("Full pickup and dropoff map")

for k_mean_point in np.array(cls_k_means.cluster_centers_):

    circle = plt.Circle((k_mean_point[0], k_mean_point[1]), 30, color='r', fill=False) 

    ax.add_artist(circle)

plt.xticks(np.arange(min(df_train['pickup_longitude']), max(df_train['dropoff_longitude'])+1, 80.0))

plt.yticks(np.arange(min(df_train['dropoff_latitude']), max(df_train['dropoff_latitude'])+1, 50.0))

plt.show()





print("For sanity check, we can check the location in real map:")

print(cls_k_means.cluster_centers_)

print("-73.97442053 40.75045692 - Midtown Manhattan")

print("40.75262546 -73.97302945 - Antarctica")

print("-1.60641994  15.84336939 - Republic of the Congo")
print("less zoom in to the center locations")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10))

ax = sns.scatterplot(x="pickup_longitude", y="pickup_latitude", hue="fare_amount" ,size="fare_amount",ax = axes[0],data=df_train)

ax = sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude",hue="fare_amount", size="fare_amount",ax = axes[0],data=df_train)

ax.set_title("Manhattan")

ax.set_xlim(-74.5,-73)

ax.set_ylim(41.5,40)



ax = sns.scatterplot(x="pickup_longitude", y="pickup_latitude", hue="fare_amount" ,size="fare_amount",ax = axes[1],data=df_train)

ax = sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude",hue="fare_amount", size="fare_amount",ax = axes[1],data=df_train)

ax.set_xlim(40.5,41)

ax.set_ylim(-73.5,-73.9)

ax.set_title("Antarctica")



ax = sns.scatterplot(x="pickup_longitude", y="pickup_latitude", hue="fare_amount" ,size="fare_amount",ax = axes[2],data=df_train)

ax = sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude",hue="fare_amount", size="fare_amount",ax = axes[2],data=df_train)

ax.set_xlim(-16.16)

ax.set_ylim(26,-24)

ax.set_title("Sea")

plt.show()
# less drop any location out of new york

df_train = df_train[(-70 >= df_train['pickup_longitude']) & (df_train['pickup_longitude'] >= -80)]

df_train = df_train[(-70 >= df_train['dropoff_longitude']) & (df_train['dropoff_longitude'] >= -80)]

df_train = df_train[(35 <= df_train['pickup_latitude']) & (df_train['pickup_latitude'] <= 45)]

df_train = df_train[(35 <= df_train['dropoff_latitude']) & (df_train['dropoff_latitude'] <= 45)]
map = folium.Map(location = [40.730610,-73.935242],zoom_start = 11.5,)



for index, row in df_train[['pickup_latitude','pickup_longitude','dropoff_longitude','dropoff_latitude']].sample(1500).iterrows():

    folium.CircleMarker([row['pickup_latitude'], row['pickup_longitude']],

                        radius=3,

                        color="#00f000", 

                        fill_opacity=0.9

                       ).add_to(map)



    

# city center 

folium.CircleMarker([40.7641244, -73.9706142],

                        radius=3,

                        color="#f00000", 

                        fill_opacity=0.9

                       ).add_to(map)

# jfk location

folium.CircleMarker([40.639722, -73.778889],

                        radius=3,

                        color="#f00000", 

                        fill_opacity=0.9

                       ).add_to(map)

# ewr location

folium.CircleMarker([40.6925, -74.168611],

                        radius=3,

                        color="#f00000", 

                        fill_opacity=0.9

                       ).add_to(map)

# lga location

folium.CircleMarker([40.77725, -73.872611],

                        radius=3,

                        color="#f00000", 

                        fill_opacity=0.9

                       ).add_to(map)



# statue of liberty

folium.CircleMarker([40.6892, -74.0445],

                        radius=3,

                        color="#f00000", 

                        fill_opacity=0.9

                       ).add_to(map)





map


def haversine(lon1, lon2, lat1, lat2):

    # Convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = radians(lon1), radians(lat1), radians(lon2), radians(lat2)



    # approximate radius of earth in km

    R = 6373.0

    

    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))



    distance = R * c

    return distance



def minkowski_distance(x1, x2, y1, y2, p):

    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
# distance from main loctions



#NY center

center_location_lon = -74.0063889

center_location_lat = 40.7641244



df_train['distance_from_pickup_to_center'] = df_train.apply(lambda row: haversine(row.pickup_longitude, center_location_lon, 

                                                row.pickup_latitude, center_location_lat), axis=1)



df_train['distance_from_dropoff_to_center'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, center_location_lon, 

                                                row.dropoff_latitude, center_location_lat), axis=1)



df_train['distance_from_pickup_to_dropoff'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, row.pickup_longitude, 

                                                row.dropoff_latitude, row.pickup_latitude), axis=1)



df_train['distance_from_center'] = df_train['distance_from_dropoff_to_center'] + df_train['distance_from_pickup_to_center'] 







ax = sns.jointplot(x="distance_from_pickup_to_center", y="fare_amount", kind="reg",data=df_train)

plt.title("distance_from_pickup_to_center Vs fare_amount")

plt.show()



ax = sns.jointplot(x="distance_from_dropoff_to_center", y="fare_amount", kind="reg",data=df_train)

plt.title("distance_from_dropoff_to_center Vs fare_amount")

plt.show()
ax = sns.jointplot(x="distance_from_pickup_to_dropoff", y="fare_amount", kind="reg",data=df_train)

plt.title("distance_from_pickup_to_dropoff Vs fare_amount")

plt.show()



ax = sns.jointplot(x="distance_from_center", y="fare_amount", kind="reg",data=df_train)

plt.title("distance_from_center Vs fare_amount")

plt.show()
# We know the airport taxi costs more than a regular taxi, so we will add the distand to the closest airport

#     JFK: John F. Kennedy International Airport

#     EWR: Newark Liberty International Airport

#     LGA: LaGuardia Airport



jfk_coord = (40.639722, -73.778889)

ewr_coord = (40.6925, -74.168611)

lga_coord = (40.77725, -73.872611)

sol_coord = (40.6892,-74.0445)



df_train['jfk_distance_pickup'] = df_train.apply(lambda row: haversine(row.pickup_longitude, jfk_coord[1], 

                                                row.pickup_latitude, ewr_coord[0]), axis=1)

df_train['jfk_distance_dropoff'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, jfk_coord[1], 

                                                row.dropoff_latitude, ewr_coord[0]), axis=1)

df_train['jfk_distance'] = df_train[['jfk_distance_pickup','jfk_distance_dropoff']].min(axis=1)





df_train['ewr_distance_pickup'] = df_train.apply(lambda row: haversine(row.pickup_longitude, ewr_coord[1], 

                                                row.pickup_latitude, ewr_coord[0]), axis=1)

df_train['ewr_distance_dropoff'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, ewr_coord[1], 

                                                row.dropoff_latitude, ewr_coord[0]), axis=1)

df_train['ewr_distance'] = df_train[['ewr_distance_pickup','ewr_distance_dropoff']].min(axis=1)





df_train['lga_distance_pickup'] = df_train.apply(lambda row: haversine(row.pickup_longitude, lga_coord[1], 

                                                row.pickup_latitude, lga_coord[0]), axis=1)

df_train['lga_distance_dropoff'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, lga_coord[1], 

                                                row.dropoff_latitude, lga_coord[0]), axis=1)

df_train['lga_distance'] = df_train[['lga_distance_pickup','lga_distance_dropoff']].min(axis=1)





df_train['sol_distance_pickup'] = df_train.apply(lambda row: haversine(row.pickup_longitude, sol_coord[1], 

                                                row.pickup_latitude, sol_coord[0]), axis=1)

df_train['sol_distance_dropoff'] = df_train.apply(lambda row: haversine(row.dropoff_longitude, sol_coord[1], 

                                                row.dropoff_latitude, sol_coord[0]), axis=1)

df_train['sol_distance'] = df_train[['sol_distance_pickup','sol_distance_dropoff']].min(axis=1)





ax = sns.jointplot(x="jfk_distance", y="fare_amount", kind="reg",data=df_train)

plt.title("jfk_distance Vs fare_amount")

plt.show()

ax = sns.jointplot(x="ewr_distance", y="fare_amount", kind="reg",data=df_train)

plt.title("ewr_distance Vs fare_amount")

plt.show()
ax = sns.jointplot(x="lga_distance", y="fare_amount", kind="reg",data=df_train)

plt.title("lga_distance Vs fare_amount")

plt.show()



ax = sns.jointplot(x="sol_distance", y="fare_amount", kind="reg",data=df_train)

plt.title("sol_distance Vs fare_amount")

plt.show()
# If we have more power full GPU , we can try to take all the zipcode from google



# for exmple: 

from geopy.geocoders import Nominatim

lan_lot_str = (df_train.pickup_latitude.astype(str)+","+df_train.pickup_longitude.astype(str))

geolocator = Nominatim(user_agent="Texi Fare Predictin")



zipcode = []

for number in lan_lot_str.sample(20):

    location = geolocator.reverse(number)

    if 'postcode' in location.raw['address'].keys():

        zipcode.append(location.raw['address']['postcode'])    

    else:

        zipcode.append(None)

    

print(zipcode)
colormap = plt.cm.RdBu

plt.figure(figsize=(20,20))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df_train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
# transformers:



class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]



# convert lat and lon to radians

class ToRadian(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    

    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        data_dict['pickup_longitude'] = np.radians(data_dict['pickup_longitude'])

        data_dict['pickup_latitude'] = np.radians(data_dict['pickup_latitude'])

        data_dict['dropoff_longitude'] = np.radians(data_dict['dropoff_longitude'])

        data_dict['dropoff_latitude'] = np.radians(data_dict['dropoff_latitude'])

        return data_dict

    

# Change pickup_datetime to day, week, month and year

class PickupTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        timedata = data_dict['pickup_datetime']

        

        day = pd.DatetimeIndex(timedata).day

        hour = pd.DatetimeIndex(timedata).hour

        week = pd.DatetimeIndex(timedata).week

        month = pd.DatetimeIndex(timedata).month

        year = pd.DatetimeIndex(timedata).year

        # convert day, hour and month to X and Y on the Unit Circle 

        data_dict['x_pickup_month']=np.sin(2.*np.pi*month/12.)

        data_dict['y_pickup_month']=np.cos(2.*np.pi*month/12.)



        dayfloat = day+hour/24.0

        data_dict['x_pickup_day']=np.sin(2.*np.pi*dayfloat/31.)

        data_dict['y_pickup_day']=np.cos(2.*np.pi*dayfloat/31.)



        data_dict['x_pickup_week']=np.sin(2.*np.pi*week/52.)

        data_dict['y_pickup_week']=np.cos(2.*np.pi*week/52.)

        

        data_dict['pickup_year'] = year

        return data_dict



# Get distance to center location 

class CenterLocationTransformer:

    def __init__(self):

        return



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        center_location_x = -73.9706142

        center_location_y = 40.7641244



        pickup_x = data_dict['pickup_longitude']

        pickup_y = data_dict['pickup_latitude']

        dropoff_x = data_dict['dropoff_longitude']

        dropoff_y = data_dict['dropoff_latitude']

        

        data_dict['distance_from_pickup_to_center'] = data_dict.apply(lambda row: haversine(row.pickup_longitude, center_location_x, 

                                                row.pickup_latitude, center_location_y), axis=1)



        data_dict['distance_from_dropoff_to_center'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, center_location_x, 

                                                row.dropoff_latitude, center_location_y), axis=1)



        data_dict['distance_from_pickup_to_dropoff'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, row.pickup_longitude, 

                                                row.dropoff_latitude, row.pickup_latitude), axis=1)

        

        data_dict['distance_from_pickup_to_center'].fillna(0,inplace = True)

        data_dict['distance_from_dropoff_to_center'].fillna(0,inplace = True)

        data_dict['distance_from_pickup_to_dropoff'].fillna(0,inplace = True)

        data_dict['distance_from_center'] = data_dict['distance_from_dropoff_to_center'] + data_dict['distance_from_pickup_to_center'] 

        return data_dict

    

# Get distance to airports

class AirportLocationTransformer:       



    def __init__(self):

        return



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        jfk_coord = (40.639722, -73.778889)

        ewr_coord = (40.6925, -74.168611)

        lga_coord = (40.77725, -73.872611)

        

        data_dict['jfk_distance_pickup'] = data_dict.apply(lambda row: haversine(row.pickup_longitude, jfk_coord[1], 

                                                row.pickup_latitude, ewr_coord[0]), axis=1)

        data_dict['jfk_distance_dropoff'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, jfk_coord[1], 

                                                row.dropoff_latitude, ewr_coord[0]), axis=1)

        data_dict['jfk_distance'] = data_dict[['jfk_distance_pickup','jfk_distance_dropoff']].min(axis=1)





        data_dict['ewr_distance_pickup'] = data_dict.apply(lambda row: haversine(row.pickup_longitude, ewr_coord[1], 

                                                row.pickup_latitude, ewr_coord[0]), axis=1)

        data_dict['ewr_distance_dropoff'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, ewr_coord[1], 

                                                row.dropoff_latitude, ewr_coord[0]), axis=1)

        data_dict['ewr_distance'] = data_dict[['ewr_distance_pickup','ewr_distance_dropoff']].min(axis=1)





        data_dict['lga_distance_pickup'] = data_dict.apply(lambda row: haversine(row.pickup_longitude, lga_coord[1], 

                                                row.pickup_latitude, lga_coord[0]), axis=1)

        data_dict['lga_distance_dropoff'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, lga_coord[1], 

                                                row.dropoff_latitude, lga_coord[0]), axis=1)

        data_dict['lga_distance'] = data_dict[['lga_distance_pickup','lga_distance_dropoff']].min(axis=1)



        

        data_dict['sol_distance_pickup'] = data_dict.apply(lambda row: haversine(row.pickup_longitude, sol_coord[1], 

                                                row.pickup_latitude, sol_coord[0]), axis=1)

        data_dict['sol_distance_dropoff'] = data_dict.apply(lambda row: haversine(row.dropoff_longitude, sol_coord[1], 

                                                row.dropoff_latitude, sol_coord[0]), axis=1)

        data_dict['sol_distance'] = data_dict[['lga_distance_pickup','lga_distance_dropoff']].min(axis=1)



        return data_dict
X = df_train.drop('fare_amount',axis = 1)

y = df_train['fare_amount']



# select features

prdict_feature = ['passenger_count','pickup_datetime','dropoff_latitude',

                  'dropoff_longitude','pickup_latitude','pickup_longitude']

X = X[prdict_feature]



# split

X_train_base, X_test_base, y_train, y_test = train_test_split(X, y)



# preper data

preper_data_pipeline = Pipeline([('time_transformer',PickupTransformer()),

                                ('center_location_transformer',CenterLocationTransformer()),

                                ('airport_location_transormer',AirportLocationTransformer())])



# dummeys

one_hot = OneHotEncoder(handle_unknown='ignore',sparse=True)

one_hot_pipeline = Pipeline([('hot_columns', ItemSelector(key = ['pickup_year'])),

                             ('oneHowEncoder', one_hot)])



# min max scaler

min_max = MinMaxScaler()

min_max_pipeline = Pipeline([('min_max_columns', ItemSelector(key = ['distance_from_dropoff_to_center','distance_from_pickup_to_center','y_pickup_week','x_pickup_week',

                                                                     'y_pickup_day','x_pickup_day','y_pickup_month','x_pickup_month', 'passenger_count',

                                                                     'dropoff_latitude','jfk_distance','ewr_distance','lga_distance',

                                                                     'dropoff_longitude','pickup_latitude','pickup_longitude','distance_from_pickup_to_dropoff'])),

                             ('minMaxScaler', min_max)])



one_hot_min_max_feature_union = FeatureUnion([('one_hot',one_hot_pipeline),

                                 ('min_max',min_max_pipeline)])



# full feature pipline

feature_pipeline = Pipeline([('preper_data',preper_data_pipeline),

                            ('one_hot_min_max_feature_union',one_hot_min_max_feature_union)])



feature_pipeline.fit(X_train_base)



X_train = feature_pipeline.transform(X_train_base)

X_test = feature_pipeline.transform(X_test_base)
reg_LR = LinearRegression()

reg_LR.fit(X_train,y_train)



LR_Score_train = np.sqrt(mean_squared_error(y_train, reg_LR.predict(X_train)))

LR_Score_test = np.sqrt(mean_squared_error(y_test, reg_LR.predict(X_test)))

LR_Cross_train = np.sqrt(cross_val_score(reg_LR,X_train, y_train, cv=5,scoring=make_scorer(mean_squared_error)).mean())



print("LinearRegression Cross train:",LR_Cross_train)

print("LinearRegression test:",LR_Score_test)
reg_SGD = SGDRegressor(max_iter=100, tol=0.01)

reg_SGD.fit(X_train,y_train)



SGD_Score_train = np.sqrt(mean_squared_error(y_train, reg_SGD.predict(X_train)))

SGD_Cross_train = np.sqrt(cross_val_score(reg_SGD,X_train, y_train, cv=5,scoring=make_scorer(mean_squared_error)).mean())

SGD_Score_test = np.sqrt(mean_squared_error(y_test, reg_SGD.predict(X_test)))



print("SGDRegressor Cross train:",SGD_Cross_train)

print("SGDRegressor test:",SGD_Score_test)
params = {

    'max_depth': 8, 

    'eta':.03,

    'subsample': 1, 

    'colsample_bytree': 0.8, 

    'objective':'reg:linear',

    'eval_metric':'rmse',

    'silent': 1

}



matrix_train = xgb.DMatrix(X_train,label=y_train)

matrix_test = xgb.DMatrix(X_test,label=y_test)

reg_xgb=xgb.train(params=params,

                    dtrain=matrix_train,num_boost_round=5000, 

                    early_stopping_rounds=50,verbose_eval=50,

                    evals=[(matrix_test,'test')])
result = xgb.cv(params=params, dtrain=matrix_train, num_boost_round=5000, early_stopping_rounds=50)

XGB_Cross_train = result['train-rmse-mean'].mean()

XGB_Score_test = np.sqrt(mean_squared_error(y_test, reg_xgb.predict(xgb.DMatrix(X_test), ntree_limit = reg_xgb.best_ntree_limit)))



print("xgb train:",XGB_Cross_train)

print("xgb test:",XGB_Score_test)
reg_DNN = Sequential([

    Dense(256, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'),

    Dense(128, kernel_initializer='normal', activation='relu'),

    Dense(64, kernel_initializer='normal', activation='relu'),

    Dense(32, kernel_initializer='normal', activation='relu'),

    Dense(1, kernel_initializer='normal')

])

optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.001)

reg_DNN.compile(loss='mean_squared_error', optimizer= optimizer)

reg_DNN.fit(X_train.toarray(), y_train, steps_per_epoch = 512,epochs=50, validation_data = (X_test.toarray(), y_test), validation_steps = 500)
DNN_Score_train = np.sqrt(mean_squared_error(y_train, reg_DNN.predict(X_train)))

DNN_Score_test = np.sqrt(mean_squared_error(y_test, reg_DNN.predict(X_test)))



print("DNN train:",DNN_Score_train)

print("DNN test:",DNN_Score_test)
score_table = pd.DataFrame([], index = ["LinearRegression","SGDRegressor", "xgboost","DNN"])

score_table["Train_Score"] = [LR_Cross_train,SGD_Cross_train,XGB_Cross_train,DNN_Score_train]

score_table["Test_Score"] = [LR_Score_test,SGD_Score_test,XGB_Score_test,DNN_Score_test]



score_table.plot.bar()

plt.show()
X_base_columns = preper_data_pipeline.fit_transform(X_train_base)

unuse_params = ['distance_from_center','sol_distance','jfk_distance_pickup','jfk_distance_dropoff','ewr_distance_pickup',

               'ewr_distance_dropoff','lga_distance_pickup','lga_distance_dropoff','sol_distance_pickup','sol_distance_dropoff','pickup_year','pickup_datetime']

base_columns = X_base_columns.columns.drop(unuse_params).values

year_columns = one_hot.categories_[0]

training_labels = np.concatenate([year_columns,base_columns])



explainer = lime.lime_tabular.LimeTabularExplainer(X_train.toarray(), feature_names=training_labels ,categorical_features = training_labels, 

                                                    verbose=True, mode='regression')



def xgb_prediction(X_array_in):

    return reg_xgb.predict(xgb.DMatrix(X_array_in), ntree_limit = reg_xgb.best_ntree_limit)





goodPrdicts = []

badPrdict = []



for i in range(1,500):

    prdict_RMSE = np.sqrt(mean_squared_error([y_test.values[i]], xgb_prediction(X_test[i])))

    if prdict_RMSE < 0.1:

        goodPrdicts.append(X_test.toarray()[i])

    elif prdict_RMSE > 10:

        badPrdict.append(X_test.toarray()[i])
exp = explainer.explain_instance(goodPrdicts[1], xgb_prediction, num_features=10)

exp.show_in_notebook(show_table=True)

exp.as_list(label=10)
exp = explainer.explain_instance(badPrdict[2], xgb_prediction, num_features=10)

exp.show_in_notebook(show_table=True)

exp.as_list(label=10)
test = feature_pipeline.transform(df_test)

predit_test = reg_xgb.predict(xgb.DMatrix(test), ntree_limit = reg_xgb.best_ntree_limit)

predit_test
#Create submission file

submission = pd.DataFrame({"key": df_test.key})

submission["fare_amount"] = predit_test.round(2)

submission



submission.to_csv('taxi_fare_submission.csv',index=False)

submission.head()

 