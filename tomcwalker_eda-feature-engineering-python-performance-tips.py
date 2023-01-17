# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# date/time maths

import datetime as datetime



# plotting tools

import matplotlib.pyplot as plt

import seaborn as sns



# general maths

import math as math
train_data = pd.read_csv("../input/train.csv", parse_dates=['pickup_datetime','dropoff_datetime'])

train_data.head()
train_data.describe(include ='all')
# Check for null entries

train_data.isnull().sum()
# Let's get some features!

# Showing that apply works - although it's slow. 

train_data['day_of_week'] = train_data['pickup_datetime'].apply(lambda dt: dt.weekday())

train_data['time_of_day'] = train_data['pickup_datetime'].apply(lambda dt: dt.time())



# WARNING - SLOW VERSION WHICH OPERATES ON DATAFRAME ROWS. INCLUDED ONLY SO YOU CAN TRY OUT 

# JUST HOW SLOW THIS IS BY COMPARISON 



# At 40° north or south*, the distance between a degree of longitude is 53 miles (85 kilometers).

# We are pretty much 40 degrees North, so 85 Km will do.

# Each degree of latitude is approximately 69 miles (111 kilometers) apart.

# ref: https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616     



LAT_SCALE_METRES = 111000

LONG_SCALE_METRES = 85000



def latToM(lat) :

    return lat * LAT_SCALE_METRES



def longToM(lng) :

    return lng * LONG_SCALE_METRES



def distanceInM( row ) :

    longDiffm = longDiffM(row)

    latDiffm = latDiffM(row)

    return np.sqrt( longDiffm * longDiffm + latDiffm * latDiffm)



def calcSpeedKmh( metres, secs) :

    # km/hour = (metres / 1000) / (secs/ (60*60) )

    return (3600 * metres) / (secs * 1000)



def speedInKmh (row) :

    return calcSpeedKmh(row['distance_in_metres'], row['trip_duration'])       



def longDiffM(row) :

    return  longToM( abs(row['dropoff_longitude'] - row['pickup_longitude']) )



def latDiffM(row) :

    return latToM( abs(row['dropoff_latitude'] - row['pickup_latitude']))



def daysFromNewYear2016(dt) :

   nyDay = datetime.date(2016,1,1)

   return (dt.date() - nyDay ).days



def direction( row ) :

    # longitude positive is e->w (as we are west of London, further west = higher value for longitude)

    # latitude is s->n so ie higher latitude is further south as we are north of eqator.

    # tangent = opposite/adjacent, ie lat dist / long dist    

    # This function returns, for example:

    # Heading westwards = 0, southwards = 1, eastwards = 2, northwards = -1.

    # Heading northwest = -0.5, northeast = -1.5, southeast = 1.5, southewst = 0.5

    # The closer the absolute value is to 1, the closer we are to going n/s

    return 2 * math.atan2( (row['dropoff_latitude'] - row['pickup_latitude']), (row['dropoff_longitude'] - row['pickup_longitude']))/math.pi



# 0 for going directly e/w or n/s

# 0.5 for a diagonal movement

# other values in between as appropriate.

# likely a more elegant way to do this

def diffFromGridDirection ( row ) :

    nsDiff = abs(abs(row['direction']) - 1)

    ewDiff = min(abs(row['direction']), abs(abs(row['direction']) - 2))

    return min(ewDiff, nsDiff)

                

            
# At 40° north or south*, the distance between a degree of longitude is 53 miles (85 kilometers).

# We are pretty much 40 degrees North, so 85 Km will do.

# Each degree of latitude is approximately 69 miles (111 kilometers) apart.

# ref: https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616     



# Vectorised version.

# See https://tomaugspurger.github.io/modern-4-performance.html for more info on why this is 

# a big speed improvements over apply





LAT_SCALE_METRES = 111000

LONG_SCALE_METRES = 85000



def latToMVec(lat) :

    return lat * LAT_SCALE_METRES



def longToMVec(lng) :

    return lng * LONG_SCALE_METRES



def distanceInMVec( long1, long2, lat1, lat2 ) :

    longDiffm = longDiffMVec(long1, long2)

    latDiffm = latDiffMVec(lat1, lat2)

    return np.sqrt( longDiffm * longDiffm + latDiffm * latDiffm)



def calcSpeedKmhVec( metres, secs) :

    # km/hour = (metres / 1000) / (secs/ (60*60) )

    return (3600 * metres) / (secs * 1000)



def speedInKmhVec ( metres, duration) :

    return calcSpeedKmh(metres, duration)       



def longDiffMVec(long1, long2) :

    return  longToMVec( abs(long2-long1) )



def latDiffMVec(lat1, lat2) :

    return latToMVec( abs(lat2-lat1))



def daysFromNewYear2016Vec(dt) :

   nyDay = datetime.date(2016,1,1)

   return [(d.date() - nyDay ).days for d in dt]



def directionVec( long1, long2, lat1, lat2 ) :

    # longitude positive is e->w (as we are west of London, further west = higher value for longitude)

    # latitude is s->n so ie higher latitude is further south as we are north of eqator.

    # tangent = opposite/adjacent, ie lat dist / long dist    

    # I've given results in degrees as it's easier to sanity check the numbers are right.

    # This function returns, for example:

    # Heading westwards = 0, southwards = 90, eastwards = 180, northwards = -90.

    # Heading northwest = -45, northeast = -135, southeast = 135, southewst = 45

    # The closer the absolute value is to 1, the closer we are to going n/s

    lngDiff = long2 - long1

    latDiff = lat2 - lat1

    atansFunc = np.vectorize( lambda y,x : 180 * math.atan2(y,x)/math.pi )

    atans = atansFunc(latDiff, lngDiff)

    #atans = pd.Series([math.atan2(y,x) for y,x in zip(latDiff, lngDiff)])

    #return 180 * atans/math.pi

    return atans

    

    

def shiftAntiClockwise( direction ) :

    newDirection = direction + 29

    if newDirection > 180 :

       newDirection = 180 - newDirection

    return newDirection





def gridDiff( direction ) :

    # Work out how many degrees a direction differs from n/s or e/w.

    # First rotate negative directions about e/w to map onto a semicircle

    diff = direction % 90

    

    if diff < 45 :

       return diff

    else :

       return 90 - diff





# 0 for going directly e/w or n/s

# 45 for a perfect diagonal.

def diffFromGridDirectionVec ( direction ) :

    # To make the maths easier, let's rotate our direction 29 degrees anticlockwise.

    # Then we check how close the result is to north/south or east/west alignment.

    

    # We use vectorise for performance, though not strictly necessary here.See:

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html

    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html    

    vFuncShift = np.vectorize( shiftAntiClockwise )

    vFuncDiff = np.vectorize( gridDiff )

    return vFuncDiff(vFuncShift(direction ))

            

   

def gridDistance ( distance, diffFromGridDirection) :

    radians = (diffFromGridDirection * math.pi)/180

    return math.sin( radians ) * distance + math.cos( radians ) * distance



def gridDistanceVec( distance, diffFromGridDirection) :

     vFuncDist = np.vectorize(gridDistance)

     return vFuncDist(distance, diffFromGridDirection)

        

        

    
train_data['distance_in_metres'] = distanceInMVec(train_data['pickup_longitude'], train_data['dropoff_longitude'], train_data['pickup_latitude'], train_data['dropoff_latitude'])

train_data['days_from_new_year'] = daysFromNewYear2016Vec(train_data['pickup_datetime'])

train_data['direction'] = directionVec(train_data['pickup_longitude'], train_data['dropoff_longitude'], train_data['pickup_latitude'], train_data['dropoff_latitude'])    

train_data['speed_in_kmh'] = speedInKmhVec(train_data['distance_in_metres'], train_data['trip_duration'])

train_data['diffFromGridDirection'] = diffFromGridDirectionVec(train_data['direction'])

train_data['grid_distance'] = gridDistanceVec( train_data['distance_in_metres'], train_data['diffFromGridDirection'] )
# Slow way, using apply

#train_data['distance_in_metres'] = train_data.apply(lambda row: distanceInM(row),axis=1)

#train_data['grid_distance_in_metres'] = train_data.apply(lambda row: gridDistanceInM(row),axis=1)

#train_data['direction'] = train_data.apply(lambda row: direction(row),axis=1)

#train_data['days_from_new_year'] = train_data['pickup_datetime'].apply(lambda dt: daysFromNewYear2016(dt))





# Slow way, using apply

#train_data['speed_in_kmh'] = train_data.apply(lambda row: speedInKmh(row),axis=1)

#train_data['grid_speed_in_kmh'] = train_data.apply(lambda row: gridSpeedInKmh(row),axis=1)

#train_data['diffFromHorVert'] = train_data.apply(lambda row: diffFromHorVert(row),axis=1)
train_data.describe()
train_data[train_data['speed_in_kmh'] > 200]
plt.figure(figsize=(12,8))

sns.distplot(train_data['speed_in_kmh'].values, bins=50, kde=True)

plt.xlabel('speed', fontsize=12)

plt.show()          
plt.figure(figsize=(12,8))

#sns.distplot(train_data['speed_in_kmh'].values, bins=50, kde=True)



sns.distplot(train_data[train_data['speed_in_kmh'] < 200]['speed_in_kmh'].values, bins = 100, kde=True)

plt.xlabel('speed', fontsize=12)

plt.show()
train_data['speed_in_kmh'].quantile(q= (0.5,0.75, 0.99, 0.995, 0.999, 0.9995, 0.9997, 0.9999, 0.99999))
train_data[ (train_data['speed_in_kmh'] > 65) & (train_data['speed_in_kmh'] < 120) & (train_data['distance_in_metres'] > 1000) ]
train_data_cleaned = train_data[train_data['speed_in_kmh'] <= 40.853]
train_data[ (train_data['trip_duration'] > 20000) ]
plt.figure(figsize=(12,8))

#sns.distplot(train_data['speed_in_kmh'].values, bins=50, kde=True)



big_trips = train_data[(train_data['trip_duration'] > 7200) ]['trip_duration']

sns.distplot(big_trips.values, bins = 100, kde=True)

plt.xlabel('trip duration when over 2 hours', fontsize=12)

plt.show()
train_data_cleaned = train_data_cleaned[train_data_cleaned['trip_duration'] < 70000]

train_data_cleaned[train_data_cleaned['trip_duration'] < 60].describe()