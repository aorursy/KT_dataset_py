#installing Tensorflow for future use 

import tensorflow as tf

tf.reset_default_graph()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
NYC_Taxi_train=pd.read_csv('../input/train.csv')

NYC_Taxi_test=pd.read_csv('../input/test.csv')
NYC_Taxi_train.head()
NYC_Taxi_train.columns
#Data fields



# id - a unique identifier for each trip

# vendor_id - a code indicating the provider associated with the trip record

# pickup_datetime - date and time when the meter was engaged

# dropoff_datetime - date and time when the meter was disengaged

# passenger_count - the number of passengers in the vehicle (driver entered value)

# pickup_longitude - the longitude where the meter was engaged

# pickup_latitude - the latitude where the meter was engaged

# dropoff_longitude - the longitude where the meter was disengaged

# dropoff_latitude - the latitude where the meter was disengaged

# store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip

# trip_duration - duration of the trip in seconds
#univariate analysis (analysis of all features individually )
# Obervation with id

# checking duplicate with 'id' feature 

NYC_Taxi_train['id'].duplicated().value_counts()

NYC_Taxi_test['id'].duplicated().value_counts()

# unique id count in train:1458644

# unique id count in test: 625134



# Obervation with vendor id

NYC_Taxi_train['vendor_id'].duplicated().value_counts()

NYC_Taxi_test['vendor_id'].duplicated().value_counts()

NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum()

# two vendor's provide taxi as per data set in future we would like to explore are they providing taxi's in some perticular let-log(area of NYC)
# popularity of vendor 

NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum().plot(kind='bar',figsize=(8,6))
# Passenger_count trend 

NYC_Taxi_train['passenger_count'].value_counts().sort_values()

NYC_Taxi_train['passenger_count'].value_counts().sort_values().plot(kind='barh',figsize=(8,6))

# Observations:

# 1) 60 taxi running with out passenger :) 

# 2) mostly passenger travel alone or with one more passenger ,after that thrid largest count is of 5 passenger's in group
# analysis of trip duration 

NYC_Taxi_train['trip_duration'].isnull().value_counts()

NYC_Taxi_train['trip_duration'].max()

NYC_Taxi_train['trip_duration'].min()

# funny minimum trip_duration is 1 sec 
NYC_Taxi_train_alt=NYC_Taxi_train[NYC_Taxi_train['trip_duration']<120]

NYC_Taxi_train_alt['trip_duration'].count()

# total trips finished with in 2 mins = 27817
# create new columns trip duration in mins AND trip duration in hours

NYC_Taxi_train['trip_duration_in_min']=(NYC_Taxi_train['trip_duration']/60).round(1)

NYC_Taxi_train['trip_duration_in_hour']=(NYC_Taxi_train['trip_duration_in_min']/60).round(2)
NYC_Taxi_train['trip_duration_in_min'].mean()
NYC_Taxi_train['trip_duration_in_min'].min()
NYC_Taxi_train['trip_duration_in_min'].max()
# in train dataset some trip duration are very high (I consider them outliers and remove them before replotting it)

q = NYC_Taxi_train.trip_duration.quantile(0.99)

NYC_Taxi_train = NYC_Taxi_train[NYC_Taxi_train.trip_duration < q]

plt.figure(figsize=(8,6))

plt.scatter(range(NYC_Taxi_train.shape[0]), np.sort(NYC_Taxi_train.trip_duration.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('trip duration', fontsize=12)

plt.show()
# lets create a copy of NYC_Taxi_train with name "temp

temp=NYC_Taxi_train.copy()
temp = temp[temp.trip_duration < temp.trip_duration.quantile(0.995)] # Temporarily removing outliers

pickup_dates = pd.DatetimeIndex(temp['pickup_datetime'])
#ow are taxi rides split among day of week and hour of day?

weekday = pickup_dates.dayofweek

day, count = np.unique(weekday, return_counts = True)



plt.figure(figsize=(6,4))

ax = sns.barplot(x = day, y = count)

ax.set(xlabel = "Day of week", ylabel = "Count of taxi rides")

plt.show();
# lets check some trand analysis regarding specific month/specific day /perticular hour of day with regards to travel duration
def toDateTime( df ):

    

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    

    df['month'] = df['pickup_datetime'].dt.month

    df['hour'] = df['pickup_datetime'].dt.hour

    df['day_week'] = df['pickup_datetime'].dt.weekday_name

    

    return df
temp1=toDateTime(temp)
temp1.head()
temp1.columns
# lets play with distance among lat-long 
def locationFeatures( df ):

    #displacement

    df['y_dis'] = df['pickup_longitude'] - df['dropoff_longitude']

    df['x_dis'] = df['pickup_latitude'] - df['dropoff_latitude']

    

    #square distance

    df['dist_sq'] = (df['y_dis'] ** 2) + (df['x_dis'] ** 2)

    

    #distance

    df['dist_sqrt'] = df['dist_sq'] ** 0.5

    

    return df
train = locationFeatures(temp)

test = locationFeatures(temp)
train.head()
# So NOW ONWARDS our focus will be train file 



# LETS ENTER IN THE WORLD OF multivariate analysis
train.groupby(['day_week']).mean()[['month','trip_duration_in_min']].round(2)
train.groupby(['day_week']).mean()[['month','trip_duration_in_min']].round(2).plot(kind='barh')
df=train.pivot_table(index='day_week',columns='month',values='trip_duration_in_min',aggfunc='mean').round(2)
df
df.plot(kind='bar',figsize=(10,10))

train.groupby('hour').mean()['trip_duration_in_min'].round(0)
train.groupby('hour').mean()['trip_duration_in_min'].round(0).plot()
df1=train.pivot_table(index='day_week',columns='hour',values='trip_duration_in_min',aggfunc='mean').round(2)
df1
df1.plot(kind='barh',figsize=(10,10))
sns.heatmap(df1.corr())

plt.figure(figsize=(15,12))
# based on above graph we can easily figure it out that 8 to 6 are pick hours for NYC
# Modeling part:

#1) data preparation for modeling 
train.columns
test.columns
train['store_and_fwd_flag'].value_counts()
lookup={'Y':'1',

        'N':'0'

       }
train['store_and_fwd_flag_1']=train['store_and_fwd_flag'].map(lookup)
train['store_and_fwd_flag_1'].value_counts()
# same thing for test

test['store_and_fwd_flag'].value_counts()

test['store_and_fwd_flag_1']=train['store_and_fwd_flag'].map(lookup)
test['store_and_fwd_flag_1'].value_counts()
train.replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],['1','2','3','4','5','6','7'],inplace=True)
trainX=train[['store_and_fwd_flag_1','dist_sqrt','month', 'hour', 'day_week','passenger_count']]

trainY=train[['trip_duration']]
def toDateTime( df ):

    

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    

    df['month'] = df['pickup_datetime'].dt.month

    df['hour'] = df['pickup_datetime'].dt.hour

    df['day_week'] = df['pickup_datetime'].dt.weekday_name

    

    return df
def locationFeatures( df ):

    #displacement

    df['y_dis'] = df['pickup_longitude'] - df['dropoff_longitude']

    df['x_dis'] = df['pickup_latitude'] - df['dropoff_latitude']

    

    #square distance

    df['dist_sq'] = (df['y_dis'] ** 2) + (df['x_dis'] ** 2)

    

    #distance

    df['dist_sqrt'] = df['dist_sq'] ** 0.5

    

    return df
train=toDateTime( NYC_Taxi_train )

test= toDateTime( NYC_Taxi_test )
train=locationFeatures( NYC_Taxi_train )

test= locationFeatures( NYC_Taxi_test )
def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    



    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
train['radial_distance'] = haversine_np(train.pickup_longitude, train.pickup_latitude,

                                           train.dropoff_longitude, train.dropoff_latitude)

test['radial_distance'] = haversine_np(test.pickup_longitude, test.pickup_latitude,

                                           train.dropoff_longitude, train.dropoff_latitude)
from sklearn.preprocessing import LabelEncoder



for f in train.columns:

    if train[f].dtype=='object':

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values)) 

        train[f] = lbl.transform(list(train[f].values))

        

train_y = train.trip_duration.values

train_X = train.drop(["id", "dropoff_datetime", "pickup_datetime", "trip_duration"], axis=1)
train_X.columns
train_X.head()
from sklearn.cross_validation import train_test_split



trainX, testX, trainY, testY = train_test_split(train_X, train_y, train_size=0.65, random_state=12345)
trainX.shape
# center and scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(train_X)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=15, random_state=1)

km.fit(X_scaled)
centroids = km.cluster_centers_

labels = km.labels_

cluster_num=15

X=X_scaled 
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D

from pylab import *

c = Counter(labels)
colors = ["g.","r.","c.","y."]



color = np.random.rand(cluster_num)



c = Counter(labels)





fig = figure()

ax = fig.gca(projection='3d')





ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2],centroids[:, 3], marker = "x", s=150, linewidths = 5, zorder = 100)



plt.show()
trainX.shape

trainY.shape

from pandas import DataFrame

trainY=DataFrame(trainY)
#Define input values

x = tf.placeholder(shape=[None,17],dtype=tf.float32, name='x-input')

y_ = tf.placeholder(shape=[None,1],dtype=tf.float32, name='y-input')



#lets normalize all features along the columns

x_n = tf.nn.l2_normalize(x,1)



print('Input placeholders created')
#Define Weights and Bias

W = tf.Variable(tf.zeros(shape=[17,1]), name="Weights")

b = tf.Variable(tf.zeros(shape=[1]),name="Bias")

print('Weight and Bias created')
#Price prediction

y = tf.add(tf.matmul(x_n,W),b,name='output')



#Loss

loss = tf.reduce_mean(tf.square(y-y_),name='Loss')



print('Output and loass Ops created')
#Lets define Gradient Descent Optimizer

train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss)



print('Optimizer is created. Graph building is completed.')
#Lets start graph Execution

with tf.Session() as sess:

    # variables need to be initialized before we can use them

    sess.run(tf.global_variables_initializer())

    

    #lets train

    training_epochs = 1000  #how many times data need to be shown to model

    

    for epoch in range(training_epochs):

        

        #Calculate train_op and loss

        train_model, train_loss = sess.run([train_op,loss],feed_dict={x:trainX, y_:trainY})

        

        if epoch % 100 == 0:

            print ('Training loss at step: ', epoch, ' is ', train_loss)

    print (sess.run([W,b]))