import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# import training data
start_time = time.time() # track time
train_df =  pd.read_csv('../input/train.csv', nrows=15000000, usecols = [1,2,3,4,5,6,7]) # the first column 'key' is not useful 
print("%s seconds" % (time.time() - start_time))

# insert testing data
start_time = time.time() # track time
test_df =  pd.read_csv('../input/test.csv', usecols = [1,2,3,4,5,6]) # the first column 'key' is not useful
print("%s seconds" % (time.time() - start_time))
# check the shape 
print (test_df.shape)
print (train_df.shape)

# check the head
print (test_df.head())
print (train_df.head())

# as expected the test data lack one column --- 'fare_amount', which is what we are going to predict --- comparing with train data
#check null value
print(test_df.isnull().sum())
#check zero value
print((test_df == 0).astype(int).sum(axis=0))
# the test data is very clean, with no null value or zero value
# check description
test_df.describe()
# By checking the description of test data, we can see the min and max value of each feature, 
# so we can choose the clean the train data base on these value. In other word, we can delete 
# the values that are out of these boundaries in the train data, as they are using in training 
# the model for prediction
#check null value 
print(train_df.isnull().sum())
#check zero value 
print((train_df == 0).astype(int).sum(axis=0))
# There are some null and zero values in the train data. This step is very import, 
# as these values can influence the training result significantly

# check description
train_df.describe()
# There are some values that are apparently to be wrong. E.g. the min of fare_amount is negative, but it can't be.
# The max value of passenger count is 208, which is too exagerating. We have to delete this values. But it doesn't 
# matter, we will delete the useless value base on the value boundary in the test data.
# One problem is the test data don't have the 'fare_amount' column, so we don't have a boundary for cleaning.
# Therefore, we want to use the visulization to see the data distribution of this paticular column value.
import matplotlib.pyplot as plt
train_df.fare_amount.hist(bins=100, figsize = (16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")

# Check quantile
train_df[['fare_amount']].quantile([0.001, 0.999])

# Most the data are distributed below 200, so check again the data distribution below 200
train_df[train_df.fare_amount <200 ].fare_amount.hist(bins=100, figsize = (16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")
# Boundary between 0 and 200 could be a good choice

# Delete null value
print("old: %d" %len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print("new: %d" %len(train_df)) # track data amount before and after deletion

# Delete zero value
print("old: %d" %len(train_df))
train_df = train_df[~(train_df == 0).any(axis=1)]
print("new: %d" %len(train_df)) # track data amount before and after deletion

# Remove observations with useless values base on the test data boundary
mask = train_df['pickup_longitude'].between(-74.3, -72.9)
mask &= train_df['dropoff_longitude'].between(-74.3, -72.9)
mask &= train_df['pickup_latitude'].between(40.5, 41.8)
mask &= train_df['dropoff_latitude'].between(40.5, 41.7)
mask &= train_df['passenger_count'].between(0, 6)
mask &= train_df['fare_amount'].between(2, 200)


print("old: %d" %len(train_df))
train_df = train_df[mask]
print("new: %d" %len(train_df)) # track data amount before and after deletion
# Strip the 'pickup_datetime' column
import datetime as dt
def convert_to_datetime(df):
    test_time = df['pickup_datetime'].astype(str).str[:-4]
    df['date_time'] =  pd.to_datetime(test_time, format='%Y%m%d %H:%M:%S')
    return df 

# Apply to both train and test data   
train_df = convert_to_datetime(train_df)
test_df = convert_to_datetime(test_df)

# Chek shape
print (test_df.shape)
print (train_df.shape)
# Extract date attributes and then drop the pickup_datetime column
def extract_date(data):
    data['hour'] = data['date_time'].dt.hour
    data['day'] = data['date_time'].dt.day
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    data['weekday'] = data['date_time'].dt.weekday
    data = data.drop(['date_time','pickup_datetime'], axis=1)
    return data

# Apply to both train and test data      
train_df = extract_date(train_df)
test_df = extract_date(test_df)

# Chek shape
print (test_df.shape)
print (train_df.shape)
# A trick for this step is we have to refer to the fare computing machanism for NYC taxi. 
# I referred to this website and apply each computing criteria to the engineering. 
# http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml

# There is no doubt that the factor that affects the fare amout the most is the distance of
# the trip, so we have the define the metric for computing distance first.
# Here is the means for computing earth surface distace base on two points' longitude and latitude.

# Define distance
def dist(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
    pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude])
    dlon = dropoff_longitude - pickup_longitude
    dlat = dropoff_latitude - pickup_latitude
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    return distance
# There are extra charges if trip ends in 3 nearby aiports and 7 nearby counties from the NYC center,
# so these location points and there distances to pickup and dropoff points are key factors 

# Distances to nearby city center, airports, and other ny counties
def transform(data):
    # Distances to nearby airports, city center and other counties
    # By reporting distances to these points, the model can somewhat triangulate other locations of interest
    
    # city center
    nyc = (-74.0060, 40.7128)
    
    # county
    Nassau = (-73.5594, 40.6546)
    Suffolk = (-72.6151, 40.9849)
    Westchester = (-73.7949, 41.1220)
    Rockland = (-73.9830, 41.1489)
    Dutchess = (-73.7478, 41.7784)
    Orange = (-74.3118, 41.3912)
    Putnam = (-73.7949, 41.4351) 

    # airport
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)
    
    
    # county
    data['pickup_distance_to_center'] = dist(nyc[0], nyc[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_center'] = dist(nyc[0], nyc[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Nassau'] = dist(Nassau[0], Nassau[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Nassau'] = dist(Nassau[0], Nassau[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Suffolk'] = dist(Suffolk[0], Suffolk[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Suffolk'] = dist(Suffolk[0], Suffolk[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Westchester'] = dist(Westchester[0], Westchester[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Westchester'] = dist(Westchester[0], Westchester[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Rockland'] = dist(Rockland[0], Rockland[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Rockland'] = dist(Rockland[0], Rockland[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Dutchess'] = dist(Dutchess[0], Dutchess[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Dutchess'] = dist(Dutchess[0], Dutchess[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Orange'] = dist(Orange[0], Orange[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Orange'] = dist(Orange[0], Orange[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_Putnam'] = dist(Putnam[0], Putnam[1],
                                      data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_Putnam'] = dist(Putnam[0], Putnam[1],
                                      data['dropoff_longitude'], data['dropoff_latitude'])
    
    # airports
    data['pickup_distance_to_jfk'] = dist(jfk[0], jfk[1],
                                         data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_jfk'] = dist(jfk[0], jfk[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_ewr'] = dist(ewr[0], ewr[1], 
                                          data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_ewr'] = dist(ewr[0], ewr[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    data['pickup_distance_to_lgr'] = dist(lgr[0], lgr[1],
                                          data['pickup_longitude'], data['pickup_latitude'])
    data['dropoff_distance_to_lgr'] = dist(lgr[0], lgr[1],
                                           data['dropoff_longitude'], data['dropoff_latitude'])
    
    # point distance
    data['distance'] = dist(data['pickup_longitude'], data['pickup_latitude'],
                            data['dropoff_longitude'], data['dropoff_latitude'])
    
    return data

# Apply to both train and test data      
train_df = transform(train_df)
test_df = transform(test_df)

# Chek shape
print (test_df.shape)
print (train_df.shape)

# Consider extra charges
def final_convert(df):

    # There is a 50-cent MTA State Surcharge for all trips that end in New York City or 
    # Nassau, Suffolk, Westchester, Rockland, Dutchess, Orange or Putnam Counties.
    # The following two variables can be merged into one.
    # The following only considers trips that starts in city center and ends in nearby counties,
    # while the opposite direction could also be considered
    # counties
    df['county_dropoff_1'] = np.where((df['pickup_distance_to_center'] <= 5) &
                                     ((df['dropoff_distance_to_Nassau'] <= 21.3) |
                                      (df['dropoff_distance_to_Westchester'] <= 22.4)), 1, 0)
    
    df['county_dropoff_2'] = np.where((df['pickup_distance_to_center'] <= 5) &                  
                                     ((df['dropoff_distance_to_Suffolk'] <= 48.7) |           
                                      (df['dropoff_distance_to_Rockland'] <= 14.1) |
                                      (df['dropoff_distance_to_Dutchess'] <= 28.7) |
                                      (df['dropoff_distance_to_Orange'] <= 29) |
                                      (df['dropoff_distance_to_Putnam'] <= 15.7)), 1, 0)
    
    # There is a daily 50-cent surcharge from 8pm to 6am.
    df['night_hour'] = np.where((df['hour'] >= 20) |
                                (df['hour'] <= 6) , 1, 0)
    
    # There is a $1 surcharge from 4pm to 8pm on weekdays, excluding holidays.
    df['peak_hour'] = np.where((df['hour'] >= 16) &
                                (df['hour'] <= 20) & 
                                (df['weekday'] >=0) &
                                (df['weekday'] <=4) , 1, 0)
    
    # This is a flat fare of $52 plus tolls, the 50-cent MTA State Surcharge, the 30-cent Improvement Surcharge, 
    # to/from JFK and any location in Manhattan:
    df['to_from_jfk'] = np.where(((df['pickup_distance_to_jfk'] <= 2) & (df['dropoff_distance_to_center'] <= 5)) | 
                                 ((df['pickup_distance_to_center'] <= 5) & (df['dropoff_distance_to_jfk'] <= 2)) ,1, 0)

    # There is a $4.50 rush hour surcharge (4 PM to 8 PM weekdays, excluding legal holidays). o/from JFK and any location in Manhattan:
    df['jfk_rush_hour'] = np.where((df['to_from_jfk'] == 1) & 
                                   (df['hour'] >= 16) &
                                   (df['hour'] <= 20) ,1, 0)
    
    # There is a $17.50 Newark Surcharge to Newark Airport:
    df['ewr'] = np.where((df['pickup_distance_to_center'] <= 5) &
                         (df['dropoff_distance_to_ewr'] <= 1) ,1, 0)
    
    return df


# Apply to both train and test data      
train_df = final_convert(train_df)
test_df = final_convert(test_df)

# Chek shape
print (test_df.shape)
print (train_df.shape)
# Check column names
print(train_df.columns)
# Check corr of 'fare_amount' to all the other variables
print(train_df.corrwith(train_df['fare_amount']))
# We can choose to remove the variables that has the corr less than 0.1
train_df = train_df.drop(['passenger_count','hour','day','month','weekday'], axis = 1)
test_df = test_df.drop(['passenger_count','hour','day','month','weekday'], axis = 1)

# Chek shape
print (test_df.shape)
print (train_df.shape)
# Split the train data for model training
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('fare_amount', axis=1),
                                                    train_df['fare_amount'], test_size=0.15, random_state = 111)

# Check shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Use lightgbm model to do the training. Need to install the lightgbm package first
# An instruction for installing can be found here:
# https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#windows

import lightgbm as lgbm
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
        'num_rounds':50000
    }


def LGBMmodel(X_train,X_test,y_train,y_test,params):
    matrix_train = lgbm.Dataset(X_train, y_train)
    matrix_test = lgbm.Dataset(X_test, y_test)
    model=lgbm.train(params=params,
                    train_set=matrix_train,
                    num_boost_round=100000, 
                    early_stopping_rounds=500,
                    verbose_eval=100,
                    valid_sets=matrix_test)
    return model
# Train the model

# model = LGBMmodel(X_train,X_test,y_train,y_test,params)

# Training RMSE best iteration is 3.25
# Predicte the 'fare_amount' and save file

# prediction = model.predict(test_df, num_iteration = model.best_iteration) 
 
# submission = pd.DataFrame(
#     {'key': test_df_initial.key, 'fare_amount': prediction},
#     columns = ['key', 'fare_amount'])
# submission.to_csv('xxx', index = False)
