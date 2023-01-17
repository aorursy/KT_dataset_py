import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from IPython.display import Image

Image('../input/nyccar/nyc.jpg')
import numpy  as np      # linear algebra

import pandas as pd      # data processing

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

#sns.set_style('darkgrid')

plt.style.use('fivethirtyeight')
train = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows = 500000)

test  = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

train = train.iloc[:,1:]

test  = test.iloc[:,1:]
print(train.dtypes)
train.head(3)
test.head(3)
print("train data size ",train.shape)

print("test  data size ",test.shape)
# Check for Null values (if any apply the appropriate method to handle them)

l = ["train set", "test set"]

j = 0

for i in [train,test]:

    print(l[j]," : \n")

    print(i.isnull().sum().sort_values(ascending = False))

    print("\n \n")

    j+=1

    
# let's drop the rows with missing values 

train.dropna(axis = 0,inplace = True)
# so we are good to go

train.isnull().sum()
# now let us see  an overview of our dataset 

train.describe().T
# drop rows with negtaive fare values .

print("Min value Before Dropping Negative values from  fare column : ",train.fare_amount.min())

train = train[train.fare_amount > 0].iloc[:,:]

print("Min value After Dropping Negative values from  fare column  : ",train.fare_amount.min())
# as you can see lat and lan values are too varying w.r.t to ,

# actual Lat and long values ranging  from -90 to 90 to -180 to 80 respectively .

print(" pickup_longitude : ",min(train.pickup_longitude),max(train.pickup_longitude))

print(" pickup_latitude  : ",min(train.pickup_latitude),max(train.pickup_latitude))

train = train[(train.pickup_longitude >  -180) & (train.pickup_longitude < 80)]

train = train[(train.pickup_latitude >  -90)   & (train.pickup_latitude < 90)]   

# as you can see lat and lan values are too varying w.r.t to ,

# actual Lat and long values ranging  from -90 to 90 to -180 to 80 respectively .

print(" dropoff_longitude : ",min(train.dropoff_longitude),max(train.dropoff_longitude))

print(" dropoff_latitude  : ",min(train.dropoff_latitude),max(train.dropoff_latitude))
train = train[(train.dropoff_longitude >  -180) & (train.dropoff_longitude < 80)]

train = train[(train.dropoff_latitude >  -90)   & (train.dropoff_latitude < 90)] 
# now for test set

train = train[(train.pickup_longitude  >  -180)  & (train.pickup_longitude < 80)]

train = train[(train.pickup_latitude   >  -90)   & (train.pickup_latitude  < 90)] 

train = train[(train.dropoff_longitude >  -180)  & (train.dropoff_longitude< 80)]

train = train[(train.dropoff_latitude  >  -90)   & (train.dropoff_latitude < 90)] 
def find_dist(slat,slong,elat,elong):

    data = [train,test]

    for i in data:

        phi1 = np.radians(i[slat])

        phi2 = np.radians(i[elat])

        delta_phi = np.radians(i[elat]-i[slat])

        delta_lambda = np.radians(i[elong]-i[slong])

        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2 

        #c = 2 * atan2( √a, √(1−a) )

        c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))

    

        d = 6371 * c

        i["distance"] = d



find_dist("pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude")
train['pickup_datetime']  = pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

test['pickup_datetime']   = pd.to_datetime(test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')


for df in [train,test]:

    df['year']  = df['pickup_datetime'].dt.year

    df['month'] = df['pickup_datetime'].dt.month

    df['date']  = df['pickup_datetime'].dt.day

    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

    df['hour']  = df['pickup_datetime'].dt.hour

    
cols = ["fare_amount","passenger_count","distance"]

plt.figure(figsize = (18,4),facecolor = "white")

plot_num = 1



for i in cols:

    if(plot_num<=3):

        ax = plt.subplot(1,3,plot_num)

        sns.distplot(train[i],color = 'red')

        plt.xlabel(i,fontsize = 15)

    plot_num+=1

        
# year v/s average fare price

# insight : fare price increasing year by year . 

pd.pivot_table(train,values = "fare_amount",index = "year",aggfunc = "mean").plot(kind = 'bar',color = 'red')

plt.show()
sns.boxplot(train['fare_amount'],color = 'blue')

plt.show()
# let's calculate interquantile range for fare_amount feature 

q1  = train['fare_amount'].quantile(0.25)

q3  = train['fare_amount'].quantile(0.75)

iqr =  q3 - q1

lb  =  q1 - 1.5*iqr

ub  =  q3 + 1.5*iqr

print("Fare Amount : ")

print("lower bound : ",lb,"upper bound : ",ub)

# but i am still i am not going to remove any outlier
#train = train[(train["fare_amount"]>0) & (train["fare_amount"] <=22.5) ].iloc[:,:]



plt.figure(figsize = (16,4),facecolor = "white")

ax = plt.subplot(1,2,1)

sns.distplot(train["fare_amount"],ax = ax,color = 'red')

plt.xlabel("fare amount distribution ",fontsize = 15)

ax = plt.subplot(1,2,2)

sns.boxplot(train["fare_amount"],ax = ax,color = 'red')

plt.xlabel("fare amount boxplot ",fontsize = 15)

plt.show()
# insight : it seems most of the cab rides took with in 10kms by the cab passengers . 

bin_0 =  train.loc[(train['distance'] == 0), ['distance']]

bin_1 =  train.loc[(train['distance']>0)&(train['distance']<=10),['distance']]

bin_2 =  train.loc[(train['distance']>10)&(train['distance']<=50),['distance']]

bin_3 =  train.loc[(train['distance']>50)&(train['distance']<=100),['distance']]

bin_4 = train.loc[(train['distance']>100)&(train['distance']<=250),['distance']]

bin_5 = train.loc[(train['distance']>250),['distance']]

bin_0['bins'] = '0'

bin_1['bins'] = '1-10'

bin_2['bins'] = '11-50'

bin_3['bins'] = '51-100'

bin_4['bins'] = '101-250'

bin_5['bins'] = '>250'

dist_bins = pd.concat([bin_0,bin_1,bin_2,bin_3,bin_4,bin_5],axis = 0)

sns.countplot(dist_bins['bins'])

plt.xlabel("Distance bins",fontsize = 15)

plt.show()
# seems no relationship between day_of_week with passengers count

sns.boxplot(y = train["passenger_count"],x = train["day_of_week"])

plt.show()
# insight :  most of the cab customers are single passengers .

plt.figure(figsize=(15,7))

sns.scatterplot(x = train["passenger_count"],y =train['fare_amount'],color = 'red')

plt.xlabel("passenger_count ",fontsize = 15)

plt.ylabel("fare_amount ",fontsize = 15)

plt.show()



# insight : fare_amount is high during 5am to 5pm ,that might be due to people used to travel during day time only.

# eg : people working in IT sector .  

plt.figure(figsize=(15,7))

plt.scatter(x=train['hour'], y=train['fare_amount'], s=2,color = 'red')

plt.xlabel("hour",fontsize = 15)

plt.ylabel("fare_amount ",fontsize = 15)

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x=train['day_of_week'], y=train['fare_amount'], s=2,color = 'red')

plt.xlabel("day_of_week",fontsize = 15)

plt.ylabel("fare_amount ",fontsize = 15)

plt.show()
train[['fare_amount','distance']].describe().T
# How many number of cancellation has been occured by the cab customer ?

cancelled_trip = train[train.distance==0].loc[:,["fare_amount","distance","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

print("cancelled_trip count :",len(cancelled_trip))

cancelled_trip.head(5)

# the reason why i am calling this as a cancelled_trip is because of same latitute and longitude values for for both pickup and dropup column . 

# and we can see below, some fare has been charged even though the distance is equal to = 0 ,this may be due to waiting charge collected from customer if they have cancelled the cab ride .
train.columns
# fare_amount is to very low even though distance is high ,

# it may due to wrong lat and long values assigned to this instances (error) which are used to calculate distance between start and end point of ride !.

train[['fare_amount','distance']].sort_values(by = "distance",axis = 0,ascending = False).head(5)
# yes did you see that !!!!

plt.figure(figsize = (16,4),facecolor = "white")

ax = plt.subplot(1,2,1)

sns.boxplot(train["distance"],ax = ax,color = 'red')

plt.xlabel("Distance (boxplot)",)

ax = plt.subplot(1,2,2)

sns.distplot(train.distance,ax = ax,color = 'red')

plt.xlabel("Distance (distribution)",)

plt.show()
# let me remove those extreme distance which are acting as outlier with the help pf IQR method .

q1,q3 = train.distance.quantile(0.25),train.distance.quantile(0.75)

IQR   = q3 - q1 

low_bound   = q1 - IQR*1.5

upper_bound = q3 + IQR*1.5

print("low_bound : ",low_bound,"upper_bound : ",upper_bound)

# distance in negetive is meaningless , 

# i am going to consider instances with distances <= 50 kms (my assumption )

# generally city taxi's are not for longer distance travel . That's a city limited !!!!(my assumption )
print("instances with distances greater than 50kms : ",len(train[train.distance>50]))

print("instances with distances lesser than  50kms : ",len(train[train.distance<50]))

train = train[train.distance<50]
# i am not going to drop pickup_datetime col from my both train and test sets .

train.drop(['pickup_datetime'],axis = 1,inplace = True)

test.drop(['pickup_datetime'],axis = 1,inplace = True)
# Correlation map

plt.figure(figsize = (16,4),facecolor = "white")

sns.heatmap(train.corr(),annot= True,cmap = 'rainbow_r',annot_kws = {"Size":14})

plt.title("Correlation Heatmap")

# insight : distance feature is highly correlated with fare_amount . 

plt.show()
# let's split data as per model requirment 

xtrain = train.drop(["fare_amount"],axis = 1)

ytrain = train["fare_amount"]

print("xtrain shape : ",xtrain.shape)

print("ytrain shape : ",ytrain.shape)

# ------------------------------------------ # 

xtest = test.iloc[:,:]

print("xtest  shape  : ",xtest.shape)
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,r2_score

from xgboost import plot_importance

model = XGBRegressor(n_estimators=100,

                    learning_rate = .1,

                    max_depth = 6,

                    random_state=42,

                    n_jobs = -1,

                    early_stopping_rounds=10)
model.fit(

    xtrain, 

    ytrain, 

    eval_metric="rmse",

    verbose=True, )
y_train_pred = model.predict(xtrain)

print("MSE  for train set : ",mean_squared_error(ytrain,y_train_pred))

print("RMSE for train set : ",mean_squared_error(ytrain,y_train_pred)**0.5)

print("r2_score for train set: ",r2_score(ytrain,y_train_pred))

r2_val = r2_score(ytrain,y_train_pred)

n = len(xtrain)

p = xtrain.shape[1]

adjusted_r2_val  = 1 - ( ((1-r2_val)*(n-1)) / (n-p-1) )

print("adjusted r2_score for train set: ",adjusted_r2_val)

sns.distplot(ytrain - y_train_pred ).set_title("error distribution between actual and predicted values")

plt.show()
# let's plot feature Importance graph

figsize=(10,10)

fig, ax = plt.subplots(1,1,figsize=figsize)

plot_importance(model,ax = ax,height = 1)

plt.show()
# let me consider first 7 features only and train our model once again so that is there any  chance we can

# observe in increase of accuracy of model .
x2_train = train[["pickup_latitude","distance","pickup_longitude","dropoff_longitude","dropoff_latitude","hour","year"]]

y2_train = train["fare_amount"]

x2_test  = test[["pickup_latitude","distance","pickup_longitude","dropoff_longitude","dropoff_latitude","hour","year"]]
model1 = XGBRegressor(n_estimators=100,

                    learning_rate = .1,

                    max_depth = 6,

                    random_state=42,

                    n_jobs = -1,

                    early_stopping_rounds=10)

model1.fit(

    x2_train, 

    y2_train, 

    eval_metric="rmse",

    verbose=True, )
y2_train_pred = model1.predict(x2_train)

print("MSE  for train set : ",mean_squared_error(y2_train,y2_train_pred))

print("RMSE for train set : ",mean_squared_error(y2_train,y2_train_pred)**0.5)

print("r2_score for train set: ",r2_score(y2_train,y2_train_pred))

r2_val = r2_score(y2_train,y2_train_pred)

n = len(x2_train)

p = x2_train.shape[1]

adjusted_r2_val  = 1 - ( ((1-r2_val)*(n-1)) / (n-p-1) )

print("adjusted r2_score for train set: ",adjusted_r2_val)

print("Accuracy not improved by !!!!!!")
sns.distplot(y2_train - y2_train_pred ).set_title("error distribution between actual and predicted values")

plt.show()
Image('../input/thankyou/j.gif')