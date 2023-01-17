# Basic Imports
import pandas as pd
import numpy as np
import urllib.request
import json
from time import sleep

# Visualization and geo-data imports
import matplotlib.pyplot as plt
import folium
from folium import plugins
import fiona
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import geopandas
from geopandas.tools import sjoin
import seaborn as sns

# Modelling and training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgbm
# Sample and visualize first 10 rows by reading part of training data

sample_train_df = pd.read_csv('../input/train.csv', nrows=10)
sample_train_df.info()
#display(sample_train_df.head())
#display(sample_train_df.tail())
del sample_train_df
%%time

### Optimize and compress certain feature's data types
#   1. float32 is enough for upto 7 digit precision (as used in GPS)
#   2. datetime object is stored as string for reading csv
#   3. passenger_count field type is changed to uint8 (not more than 256 passenger count in training data)
#   4. Ignoring key object field

train_df_type = {
    'fare_amount':'float32',
    'pickup_datetime':'str',
    'pickup_longitude':'float32',
    'pickup_latitude':'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}

# Read training csv in chunks and storing chunks in single list

chunksize=10**7
df_train_list=[]
cnt=1
for chunk in pd.read_csv('../input/train.csv', dtype=train_df_type, usecols=list(train_df_type), chunksize=chunksize):
    # converting pickup_datetime from string to datetime object
    chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'].str.slice(0,19), utc=True, format='%Y-%m-%d %H:%M:%S')
    df_train_list.append(chunk)
    print(cnt, "chunk appended")
    cnt+=1
    
# concatenating list of training data to single dataframe

train_df = pd.concat(df_train_list)
del df_train_list
train_df.info()
# Describe training data statistics
train_df.describe()
# read test csv and describe its statistics

test_df = pd.read_csv('../input/test.csv')
test_df.describe()
### Observations from description of training and testing data:
#   1. Minimum fare amount in training data is negative
#   2. Maximum count of passenger in test data is 6

print("Instances where features of training data is null: ")
display(train_df.isnull().sum()) # any null values in train data

print("Occurences of negative fare amount: " + str(len(train_df[train_df['fare_amount']<0]))) # fare amount is zero or negative
print("Occurences with more than 6 passengers: " + str(len(train_df[train_df['passenger_count']>6]))) # passenger count is more than 6
print("Occurences with exactly 0 passengers: " + str(len(train_df[train_df['passenger_count']==0])))
def getEuclidean_distance(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    return np.sqrt(((pickup_long-dropoff_long)**2) + ((pickup_lat-dropoff_lat)**2))

def get_manhattan_dist(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    return ((dropoff_long - pickup_long).abs() + (dropoff_lat - pickup_lat).abs())

%%time

# Add new column (manhattan_dist) to both training and test dataframe.

train_df['manhattan_dist'] = get_manhattan_dist(train_df.pickup_longitude, train_df.pickup_latitude,
                                              train_df.dropoff_longitude, train_df.dropoff_latitude).astype(np.float32)

test_df['manhattan_dist'] = get_manhattan_dist(test_df.pickup_longitude, test_df.pickup_latitude,
                                       test_df.dropoff_longitude, test_df.dropoff_latitude).astype(np.float32)
def clean_data(train_df):
    
    print("Initial Train dataframe length: " + str(len(train_df)))
    
    # Remove null data
    train_df=train_df.dropna(how='any',axis='rows')
    print("Train dataframe length after removing NULL values: " + str(len(train_df)))
    
    train_df=train_df[(train_df.passenger_count<=6) & (train_df.passenger_count>=1)]
    print("Train dataframe length after filtering based on passenger counts: " + str(len(train_df)))
    
    train_df=train_df[(train_df.fare_amount>0) | ((train_df.fare_amount==0) & (train_df.manhattan_dist<0.75))]
    train_df=train_df[(train_df.fare_amount <= train_df.fare_amount.mean()+10*train_df.fare_amount.std())]
    print("Train dataframe length after filtering based on fare amount: " + str(len(train_df)))
    
    train_df=train_df[(train_df.pickup_longitude>=min(test_df.pickup_longitude)) & (train_df.pickup_longitude<=max(test_df.pickup_longitude))]
    train_df=train_df[(train_df.pickup_latitude>=min(test_df.pickup_latitude)) & (train_df.pickup_latitude<=max(test_df.pickup_latitude))]    
    train_df=train_df[(train_df.dropoff_longitude>=min(test_df.dropoff_longitude)) & (train_df.dropoff_longitude<=max(test_df.dropoff_longitude))]
    train_df=train_df[(train_df.dropoff_latitude>=min(test_df.dropoff_latitude)) & (train_df.dropoff_latitude<=max(test_df.dropoff_latitude))]
    print("Train dataframe length after filtering based on test data coordinates boundary: " + str(len(train_df)))
    
    return train_df



train_df=clean_data(train_df)
def getAirport_pickup_distance(dropoff_long, dropoff_lat, airport):
    pickup_long=0
    pickup_lat=0
    if airport == "JFK":
        pickup_long=-73.7822222222
        pickup_lat=40.6441666667
    elif airport == "LGA":
        pickup_long=-73.87
        pickup_lat=40.77
    elif airport == "EWR":
        pickup_long=-74.175
        pickup_lat=40.69
    return np.sqrt(((pickup_long-dropoff_long)**2) + ((pickup_lat-dropoff_lat)**2))

def getAirport_dropoff_distance(pickup_long, pickup_lat, airport):
    dropoff_long=0
    dropoff_lat=0
    if airport == "JFK":
        dropoff_long=-73.7822222222
        dropoff_lat=40.6441666667
    elif airport == "LGA":
        dropoff_long=-73.87
        dropoff_lat=40.77
    elif airport == "EWR":
        dropoff_long=-74.175
        dropoff_lat=40.69
    return  np.sqrt(((pickup_long-dropoff_long)**2) + ((pickup_lat-dropoff_lat)**2))

%%time

train_df['jfk_pickup_dist']=getAirport_pickup_distance(train_df.dropoff_longitude, train_df.dropoff_latitude, "JFK").astype(np.float32)
train_df['lga_pickup_dist']=getAirport_pickup_distance(train_df.dropoff_longitude, train_df.dropoff_latitude, "LGA").astype(np.float32)
train_df['ewr_pickup_dist']=getAirport_pickup_distance(train_df.dropoff_longitude, train_df.dropoff_latitude, "EWR").astype(np.float32)

train_df['jfk_dropoff_dist']=getAirport_dropoff_distance(train_df.pickup_longitude, train_df.pickup_latitude, "JFK").astype(np.float32)
train_df['lga_dropoff_dist']=getAirport_dropoff_distance(train_df.pickup_longitude, train_df.pickup_latitude, "LGA").astype(np.float32)
train_df['ewr_dropoff_dist']=getAirport_dropoff_distance(train_df.pickup_longitude, train_df.pickup_latitude, "EWR").astype(np.float32)

test_df['jfk_pickup_dist']=getAirport_pickup_distance(test_df.dropoff_longitude, test_df.dropoff_latitude, "JFK").astype(np.float32)
test_df['lga_pickup_dist']=getAirport_pickup_distance(test_df.dropoff_longitude, test_df.dropoff_latitude, "LGA").astype(np.float32)
test_df['ewr_pickup_dist']=getAirport_pickup_distance(test_df.dropoff_longitude, test_df.dropoff_latitude, "EWR").astype(np.float32)

test_df['jfk_dropoff_dist']=getAirport_dropoff_distance(test_df.pickup_longitude, test_df.pickup_latitude, "JFK").astype(np.float32)
test_df['lga_dropoff_dist']=getAirport_dropoff_distance(test_df.pickup_longitude, test_df.pickup_latitude, "LGA").astype(np.float32)
test_df['ewr_dropoff_dist']=getAirport_dropoff_distance(test_df.pickup_longitude, test_df.pickup_latitude, "EWR").astype(np.float32)
train_df['euclidean_dist'] = getEuclidean_distance(train_df.pickup_longitude, train_df.pickup_latitude,
                                                   train_df.dropoff_longitude, train_df.dropoff_latitude).astype(np.float32)

test_df['euclidean_dist'] = getEuclidean_distance(test_df.pickup_longitude, test_df.pickup_latitude,
                                       test_df.dropoff_longitude, test_df.dropoff_latitude).astype(np.float32)
%%time
def haversine_np(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2 = map(np.radians, [lon1,lat1,lon2,lat2])
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c=2*np.arcsin(np.sqrt(a))
    km=6367*c
    return km

train_df['haversine_dist']=haversine_np(train_df.pickup_longitude, train_df.pickup_latitude, train_df.dropoff_longitude, train_df.dropoff_latitude).astype(np.float32)
test_df['haversine_dist']=haversine_np(test_df.pickup_longitude, test_df.pickup_latitude, test_df.dropoff_longitude, test_df.dropoff_latitude).astype(np.float32)
%%time
# saving training dataframe to feather file
train_df=train_df.reset_index(drop=True)
train_df.to_feather('train.feather')
%%time

# read from feather file (not from original train csv) for fast loading 

train_df = pd.read_feather('train.feather')
train_df.info()
%%time
# saving training dataframe to feather file
test_df=test_df.reset_index(drop=True)
test_df.to_feather('test.feather')
%%time

# read from feather file (not from original train csv) for fast loading 

test_df = pd.read_feather('test.feather')
test_df.info()
# Haversine Correlation
train_df.haversine_dist.corr(train_df.fare_amount)
# Manhattan Correlation
train_df.manhattan_dist.corr(train_df.fare_amount)
# Euclidean Correlation
train_df.euclidean_dist.corr(train_df.fare_amount)
%%time

time_of_day = train_df.pickup_datetime.dt.hour * 3600 + train_df.pickup_datetime.dt.minute * 60 + train_df.pickup_datetime.dt.second
# time of day and distance traveled correlation
time_of_day.corr(train_df.euclidean_dist)
# time of day and the taxi fare correlation
time_of_day.corr(train_df.fare_amount)
plt.figure(figsize=(15,7))
plt.scatter(train_df.euclidean_dist[0:10**3],train_df.fare_amount[0:10**3])
plt.title('Distance vs Fare')
plt.xlabel('Distances')
plt.ylabel('Fare')
plt.figure(figsize=(15,7))
plt.scatter(train_df.fare_amount[0:10**3],train_df.euclidean_dist[0:10**3])
plt.title('Fare vs Distance')
plt.xlabel('Fare')
plt.ylabel('Distances')
plt.figure(figsize=(15,7))
plt.scatter(time_of_day[0:10**3],train_df.euclidean_dist[0:10**3])
plt.title('Time vs Distance')
plt.xlabel('Seconds elapsed from midnight')
plt.ylabel('Distances')
plt.figure(figsize=(15,7))
plt.scatter(time_of_day[0:10**3],train_df.fare_amount[0:10**3])
plt.title('Time vs Fare')
plt.xlabel('Seconds elapsed from midnight')
plt.ylabel('Fare')
del time_of_day # free up bit memory
plt.figure(figsize=(15,7))
plt.hist(train_df.pickup_datetime.dt.hour, bins=100)
plt.title('Hour vs Number of Rides')
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=train_df.pickup_datetime[0:10**6].dt.hour, y=train_df['fare_amount'][0:10**6], s=1.5)
plt.title('Hour vs Fare')
plt.xlabel('Hour')
plt.ylabel('Fare')
# initialize map with first row from training data as coordinates
m = folium.Map([40.721317, -73.844315], zoom_start=11)

for index, row in train_df[0:1000].iterrows():
    folium.CircleMarker([row['pickup_latitude'], row['pickup_longitude']],
                        radius=0.00001,
                        fill_color="#3db7e4"
                       ).add_to(m)
    
# convert to (n, 2) nd-array format for heatmap
stationArr = train_df[0:2500][['pickup_latitude', 'pickup_longitude']].as_matrix()

# plot heatmap
m.add_child(plugins.HeatMap(stationArr, radius=15))
m
del stationArr
del m
m = folium.Map([40.712276, -73.841614], zoom_start=11)

for index, row in train_df[0:1000].iterrows():
    folium.CircleMarker([row['dropoff_latitude'], row['dropoff_longitude']],
                        radius=0.00001,
                        fill_color="#3db7e4"
                       ).add_to(m)
    
# convert to (n, 2) nd-array format for heatmap
stationArr = train_df[0:2500][['dropoff_latitude', 'dropoff_longitude']].as_matrix()

# plot heatmap
m.add_child(plugins.HeatMap(stationArr, radius=15))
m
del stationArr
del m
train_df=train_df[0:10**5]
train_df=train_df.drop(['manhattan_dist','haversine_dist'],axis=1)
# Optimized method to compute whether point is inside any of the polygons or not
'''
%%time

tstPT = train_df.apply(lambda x: Point((float(x.pickup_longitude), float(x.pickup_latitude))), axis=1)
crs = {'init': 'epsg:27700'}
gdf = geopandas.GeoDataFrame(train_df, crs=crs, geometry = tstPT)

pointInPolys = sjoin(gdf, poly, how='left')
pickup_grouped = pointInPolys.groupby('index_right').groups

tstPT = train_df.apply(lambda x: Point((float(x.dropoff_longitude), float(x.dropoff_latitude))), axis=1)
crs = {'init': 'epsg:27700'}
gdf = geopandas.GeoDataFrame(train_df, crs=crs, geometry = tstPT)

pointInPolys = sjoin(gdf, poly, how='left')
dropoff_grouped = pointInPolys.groupby('index_right').groups
#display(len(dropoff_grouped[0.0]))

###
across_borough=[]
for ind, row in train_df.iterrows():
    in_bronx=(ind in pickup_grouped[1.0]) and (ind in dropoff_grouped[1.0])
    in_staten=(ind in pickup_grouped[2.0]) and (ind in dropoff_grouped[2.0])
    in_brooklyn=(ind in pickup_grouped[3.0]) and (ind in dropoff_grouped[3.0])
    in_queens=(ind in pickup_grouped[4.0]) and (ind in dropoff_grouped[4.0])
    in_manhattan=(ind in pickup_grouped[0.0]) and (ind in dropoff_grouped[0.0])
    if in_manhattan==True or in_bronx==True or in_staten==True or in_brooklyn==True or in_queens==True:
        across_borough.append(0)
    else:
        across_borough.append(1)
        
###
train_df['across_borough']=across_borough 

###
del tstPT
del crs
del gdf
del pointInPolys
del pickup_grouped
del dropoff_grouped

print("Number of rides across borough")
len(train_df[train_df.across_borough==1])
'''
%%time
# saving training dataframe to feather file
train_df=train_df.reset_index(drop=True)
train_df.to_feather('final_train.feather')
test_df=test_df.drop(['manhattan_dist','haversine_dist'],axis=1)
%%time
# saving training dataframe to feather file
test_df=test_df.reset_index(drop=True)
test_df.to_feather('final_test.feather')
%%time

# read from feather file (not from original train csv) for fast loading 

train_df = pd.read_feather('final_train.feather')
train_df.info()
%%time

# read from feather file (not from original train csv) for fast loading 

test_df = pd.read_feather('final_test.feather')
test_df.info()
f, ax = plt.subplots(figsize=(10, 8))
corr = train_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
train_df.corr(method='pearson', min_periods=1)
key = test_df['key']
test_df = test_df.drop(['key','pickup_datetime'],axis=1)
y_train_fare=train_df['fare_amount']
train_df=train_df.drop(['fare_amount','pickup_datetime'],axis=1)
### Drop any more features if you want

#test_df = test_df.drop(['passenger_count','lga_pickup_dist','lga_dropoff_dist','actual_dist'],axis=1)

#train_df=train_df.drop(['passenger_count','lga_pickup_dist','lga_dropoff_dist','actual_dist'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_df,y_train_fare, test_size=0.01)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_train,y_train))
print(lm.score(X_test,y_test))
train_df.info()

print('Intercept', round(lm.intercept_, 4))

print('pickup_longitude coef: ', round(lm.coef_[0], 4),
      '\npickup_latitude coef:', round(lm.coef_[1], 4), 
      '\ndropoff_longitude coef:', round(lm.coef_[2], 4),
      '\ndropoff_latitude coef:', round(lm.coef_[3], 4), 
      '\npassenger_count coef:', round(lm.coef_[4], 4), 
      '\njfk_pickup_dist coef:', round(lm.coef_[5], 4), 
      '\nlga_pickup_dist coef:', round(lm.coef_[6], 4), 
      '\newr_pickup_dist coef:', round(lm.coef_[7], 4), 
      '\njfk_dropoff_dist coef:', round(lm.coef_[8], 4), 
      '\nlga_dropoff_dist coef:', round(lm.coef_[9], 4), 
      '\newr_dropoff_dist coef:', round(lm.coef_[10], 4), 
      '\neuclidean_dist coef:', round(lm.coef_[11], 4), 
      #'\ncross_borough coef:', round(lm.coef_[12], 4), 
      #'\nactual_dist coef:', round(lm.coef_[13], 4)
     )
y_pred = lm.predict(train_df)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y_train_fare))
lrmse
LinearPredictions = lm.predict(test_df)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions

linear_submission = pd.DataFrame({"key": key,"fare_amount": LinearPredictions},columns = ['key','fare_amount'])
linear_submission.to_csv('submission.csv', index = False)
dtrain = xgb.DMatrix(train_df, label=y_train_fare)
dtest = xgb.DMatrix(test_df)
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.1
         }
num_rounds = 50
xb = xgb.train(params, dtrain, num_rounds)
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)
xgb_submission = pd.DataFrame({"key": key,"fare_amount": y_pred_xgb},columns = ['key','fare_amount'])
xgb_submission.to_csv('submission.csv', index = False)
rf = RandomForestRegressor()
rf.fit(train_df,y_train_fare)
rf_predict = rf.predict(test_df)
y_pred = rf.predict(train_df)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y_train_fare))
lrmse
rf_predict
rf_submission = pd.DataFrame({"key": key,"fare_amount": rf_predict},columns = ['key','fare_amount'])
rf_submission.to_csv('submission.csv', index = False)
