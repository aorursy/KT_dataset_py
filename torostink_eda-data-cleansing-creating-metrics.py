import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

import datetime as dt

from geopy.distance import vincenty, great_circle

df=pd.read_csv("../input/train.csv")

df.head()
df['duration_hr']=df['trip_duration']/3600

df['pickup_datetime']=pd.to_datetime(df.pickup_datetime)

df['dropoff_datetime']=pd.to_datetime(df.dropoff_datetime)

df['date']=df['pickup_datetime'].dt.date

df['hour']=df['pickup_datetime'].dt.hour
#Testing Geopy vincenty formula with current data

start=(df['pickup_latitude'][1],df['pickup_longitude'][1])

stop=(df['dropoff_latitude'][1],df['dropoff_longitude'][1])

print (vincenty((40.738563537597656, -73.980415344238267), (40.731151580810547, -73.999481201171875)).miles)

#for i in len(df):

#print(start, stop)

#Creating a column of distances from pickup and drop off based on longitude and lat

distance=[]

for i in range(len(df)):

    dist=(vincenty((df['pickup_latitude'][i],df['pickup_longitude'][i]),(df['dropoff_latitude'][i],df['dropoff_longitude'][i])).miles)

    distance.append(dist)

df['distance_mi']=distance

#lets also create a metric of average mph for the trip

df['mph']=df['distance_mi']/df['duration_hr']



#Lets do some further EDA, by visualizing distances in the data set

plt.scatter(range(len(df)),df.distance_mi)

plt.show()

#I'm highly doubtful that anyone took a cab that drove on average > 80 mph  Lets look at those as a subset

df_highspeed=df[df.mph >80]

#lets look at these

df_highspeed
#Assuming that the trip duration value is more accurate than long and lat values (due to reliance on gps) any trip where the average speed was >80 mph most likely has a bad value in the coordinates

#lets remove these suspect values from the dataset

df_clean_speed=df[~df.index.isin(df_highspeed.index)]
#quick check that we eliminated those 129 rows

print(len(df)-len(df_clean_speed))
#lets now take a naive estimate of average mph based on partially cleaned dataset

average_mph=df_clean_speed.mph.mean()