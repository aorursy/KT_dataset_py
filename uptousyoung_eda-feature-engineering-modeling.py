import numpy as np 

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns

% matplotlib inline

import os

print(os.listdir("../input"))
# read data in pandas dataframe

df_train =  pd.read_csv('../input/train.csv',nrows= 5000000,parse_dates=["pickup_datetime"])



# list first few rows (datapoints)

df_train.head()
df_train.columns.values
df_train.describe()
df_train.dtypes
df_train.isnull().sum()
df_train = df_train.dropna()
#calculate trip distance in miles

def distance(lat1, lat2, lon1,lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
df_train['trip_distance']=df_train.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)
df_train['trip_distance'].head(10)


# Cab rides should not have negative numbers, along with that, taxi standarad fares begin at $2.50

df_train = df_train[df_train['fare_amount'] >= 2.5]

    

# our latitude and longitude should not be equal to 0 becuase the dataset is based in NY

df_train = df_train[df_train['pickup_latitude']!= 0]

df_train = df_train[df_train['pickup_longitude'] != 0]

df_train = df_train[df_train['dropoff_latitude'] != 0]

df_train = df_train[df_train['dropoff_longitude'] != 0]



# latitude and longitude are bounded by 90 and -90. We shouldnt have any coordiantes out of that range

df_train = df_train[(df_train['pickup_latitude']<=90) & (df_train['pickup_latitude']>=-90)]

df_train = df_train[(df_train['pickup_longitude']<=90) & (df_train['pickup_longitude']>=-90)]

df_train = df_train[(df_train['dropoff_latitude']<=90) & (df_train['dropoff_latitude']>=-90)]

df_train = df_train[(df_train['dropoff_longitude']<=90) & (df_train['dropoff_longitude']>=-90)]

    

# I dont want to include destinations that have not moved from there pickup coordinates to there dropoff coordinates

df_train = df_train[(df_train['pickup_latitude'] != df_train['dropoff_latitude']) & (df_train['pickup_longitude'] != df_train['dropoff_longitude'])]
# list first few rows (datapoints)

df_train.head()
sns.distplot(df_train['fare_amount'])



plt.title('Distribution of Fare Amount')
%matplotlib inline

sns.boxplot(df_train['fare_amount'], palette="Set2" )



plt.title('Looking for Outliers with a Boxplot')
df_train = df_train[(df_train['fare_amount'] >= 2.5) & (df_train['fare_amount'] <= 800) ]
mean = np.mean(df_train['fare_amount'])

sd = np.std(df_train['fare_amount'])    

df_train = df_train[(df_train['fare_amount'] > mean - 3*sd) & (df_train['fare_amount'] < mean + 3*sd)]
sns.distplot(df_train['fare_amount'])



plt.title('Distribution of Fare Amount after removing outliers')
len(df_train)
# Double check the coordinate by adding new features 'abs_diff_longitude' and 'abs_diff_latitude'

def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(df_train)
plot = df_train.iloc[:4714682].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(df_train))

df_train = df_train[(df_train.abs_diff_longitude < 5.0) & (df_train.abs_diff_latitude < 5.0)]

print('New size: %d' % len(df_train))
# Check the distribution of "passenger_count"

passenger_count = df_train.groupby(['passenger_count']).count()



fig, ax = plt.subplots(figsize=(15,8))



sns.barplot(passenger_count.index, passenger_count['key'], palette = "Set3")



plt.xlabel('Number of Passengers')

plt.ylabel('Count')

plt.title('Count of Passengers')

plt.show()
passenger_fare = df_train.groupby(['passenger_count']).mean()



fig, ax = plt.subplots(figsize=(15,8))



sns.barplot(passenger_fare.index, passenger_fare['fare_amount'], palette = "Set3")



plt.xlabel('Number of Passengers')

plt.ylabel('Average Fare Price')

plt.title('Average Fare Price for Number of Passengers')

plt.show()
df_train = df_train[(df_train['passenger_count']<=7) & (df_train['passenger_count']>=1)]
import folium


coordinates = [[40.711303, -74.016048],[40.782004, -73.979268],]



# Create the map and add the line

m = folium.Map(location=[40.730610,-73.935242], zoom_start=12)

my_PolyLine=folium.PolyLine(locations=coordinates,weight=5, color = "black")

m.add_children(my_PolyLine)







# Add some useful features instead of the feature "key" (Date)

def date_columns(data):

    data['key'] = pd.to_datetime(data['key'], yearfirst=True)

    data['year'] = data['key'].dt.year

    data['month'] = data['key'].dt.month

    data['day'] = data['key'].dt.day

    data['weekday'] = data['key'].dt.weekday

    data['hour'] = data['key'].dt.hour

    #data['day_of_week'] = data['key'].dt.day_name()
date_columns(df_train)

df_train.columns.values
df_train.dtypes
df_train.head(10)
#Hours_Plot

time_of_day = df_train.groupby(['hour']).mean()



plt.figure(figsize=(20,8))

plt.plot(time_of_day.index, time_of_day.fare_amount, color = 'blue')



plt.xlabel('Hour')

plt.ylabel('Fare Price')

plt.title('Average Fare Price During Time of Day')

plt.show()
#Time Series Plot



taxi = df_train.sort_values(by='key').reset_index()



year = taxi['key'].dt.year.astype(str)

month = taxi['key'].dt.month.astype(str)

day = taxi['key'].dt.day.astype(str)



date = year+"-"+month+"-"+day

date = pd.to_datetime(date)

year_month = year +'-'+month

year_month = pd.to_datetime(year_month)

taxi['year_month'] = year_month

taxi['date'] = date





taxi_rate = taxi.groupby(['date']).mean()







plt.figure(figsize=(20,8))



plt.plot(taxi_rate.index, taxi_rate.fare_amount, color = "#C2A0FA")







plt.xlabel('Year')

plt.ylabel('Average Fare Price Per Day')

plt.title('Average Fare Price Over the Years')

plt.show()
df_train = df_train[(df_train['year']>=2013)]



df_train.sort_values(by='year',ascending= True)
from  sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score 

from sklearn.linear_model import LinearRegression 

import xgboost as xgb
import statsmodels.api as sm # import statsmodels 

X = df_train.drop(['fare_amount','key', 'pickup_datetime'],axis = 1)

y = df_train['fare_amount']

X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model



# Note the difference in argument order

model = sm.OLS(y, X).fit() ## sm.OLS(output, input)

predictions = model.predict(X)



# Print out the statistics

model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)


lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.score(X_train,y_train))

print(lm.score(X_test,y_test))
y_pred = lm.predict(X_test)

lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y_test))

lrmse
print ('Score:', lm.fit(X_train,y_train).score(X_test, y_test))
from sklearn.ensemble import RandomForestRegressor



randomForest = RandomForestRegressor(random_state=42)

randomForest.fit(X_train, y_train)
randomForestPredict = randomForest.predict(X_test)

randomForest_mse = mean_squared_error(y_test, randomForestPredict)

randomForestMSE = np.sqrt(randomForest_mse)

randomForestMSE
from sklearn.ensemble import GradientBoostingRegressor

model_gradient= GradientBoostingRegressor(n_estimators=100, learning_rate=1, max_depth=3, random_state=0)

gradientBoost = model_gradient.fit(X_train, y_train)
predicted = model_gradient.predict(X_test)

grmse = np.sqrt(metrics.mean_squared_error(predicted, y_test))

grmse
regression = pd.DataFrame({"regression": ['Multi Linear Regression','Random Forest',  'Gradient Boosting Regrssion'], 

                           "rmse": [lrmse,randomForestMSE,grmse]},columns = ['regression','rmse'])
regression = regression.sort_values(by='rmse', ascending = False)
sns.barplot(regression['rmse'], regression['regression'], palette = 'Set2')

plt.xlabel("Root Mean Square Error")

plt.ylabel('Regression Type')

plt.title('Comparing the different types of Regressions used')