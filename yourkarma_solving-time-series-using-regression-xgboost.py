# import libraries

import numpy as np

import pandas as pd

import time

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



from xgboost import XGBRegressor

from xgboost import plot_importance



# Function to plot feature importance

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

import matplotlib.pyplot as plt

# Read train , test and submission csv in pandas dataframe

train=pd.read_csv('/kaggle/input/train.csv',parse_dates=['DateTime'])

test=pd.read_csv('/kaggle/input/test.csv',parse_dates=['DateTime'])

sub=pd.read_csv('/kaggle/input/sub.csv')
# let's check first 5 rows from train 

train.head()
# let's check last 5 rows from train 

train.tail()
# let's explore test data's first 5 and last 5 row

test.head().append(test.tail())
train.loc[:,['DateTime','Vehicles']].plot(x='DateTime',y='Vehicles',title='Vehicle Trend',figsize=(16,4))
# filtering data greater than or equal to 01 Jan 2016

train=train[train['DateTime']>='2016-01-01']
# concat train, test data and mark where it is test , train 

train['train_or_test']='train'

test['train_or_test']='test'

df=pd.concat([train,test])
# Below function extracts date related features from datetime

def create_date_featues(df):



    df['Year'] = pd.to_datetime(df['DateTime']).dt.year



    df['Month'] = pd.to_datetime(df['DateTime']).dt.month



    df['Day'] = pd.to_datetime(df['DateTime']).dt.day



    df['Dayofweek'] = pd.to_datetime(df['DateTime']).dt.dayofweek



    df['DayOfyear'] = pd.to_datetime(df['DateTime']).dt.dayofyear



    df['Week'] = pd.to_datetime(df['DateTime']).dt.week



    df['Quarter'] = pd.to_datetime(df['DateTime']).dt.quarter 



    df['Is_month_start'] = pd.to_datetime(df['DateTime']).dt.is_month_start



    df['Is_month_end'] = pd.to_datetime(df['DateTime']).dt.is_month_end



    df['Is_quarter_start'] = pd.to_datetime(df['DateTime']).dt.is_quarter_start



    df['Is_quarter_end'] = pd.to_datetime(df['DateTime']).dt.is_quarter_end



    df['Is_year_start'] = pd.to_datetime(df['DateTime']).dt.is_year_start



    df['Is_year_end'] = pd.to_datetime(df['DateTime']).dt.is_year_end



    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)



    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)



    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)

    

    df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour

    

    return df
# extracting time related 

df=create_date_featues(df)
for col in ['Junction']:

    df = pd.get_dummies(df, columns=[col])
train=df.loc[df.train_or_test.isin(['train'])]

test=df.loc[df.train_or_test.isin(['test'])]

train.drop(columns={'train_or_test'},axis=1,inplace=True)

test.drop(columns={'train_or_test'},axis=1,inplace=True)
train['Vehicles']=np.log1p(train['Vehicles'])
train1=train[train['DateTime']<'2017-03-01']#Train period from 2016-01-01 to 2017-02-31

val1=train[train['DateTime']>='2017-03-01'] #Month 3,4,5,6 as validtaion period
def datetounix(df):

    # Initialising unixtime list

    unixtime = []

    

    # Running a loop for converting Date to seconds

    for date in df['DateTime']:

        unixtime.append(time.mktime(date.timetuple()))

    

    # Replacing Date with unixtime list

    df['DateTime'] = unixtime

    return(df)

train1=datetounix(train1)

val1=datetounix(val1)



train=datetounix(train)

test=datetounix(test)
x_train1=train1.drop(columns={'ID','Vehicles'},axis=1)

y_train1=train1.loc[:,['Vehicles']]



x_val1=val1.drop(columns={'ID','Vehicles'},axis=1)

y_val1=val1.loc[:,['Vehicles']]
ts = time.time()



model = XGBRegressor(

    max_depth=8,

    booster = "gbtree",

    n_estimators=100000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,

    seed=42,

    objective='reg:linear')



model.fit(

    x_train1, 

    y_train1, 

    eval_metric="rmse", 

    eval_set=[(x_train1, y_train1), (x_val1, y_val1)], 

    verbose=True, 

    early_stopping_rounds = 100)



time.time() - ts
#predicting validation data.

pred=model.predict(x_val1)
from sklearn.metrics import mean_squared_error

from math import sqrt

np.sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred)))
import matplotlib.pyplot as plt

%matplotlib inline

plot_features(model, (10,14))
#checks error in prediction

res = pd.DataFrame(data = pd.concat([x_val1,y_val1],axis=1))

res['Prediction']= np.expm1(model.predict(x_val1))

res['Ratio'] = res.Prediction/np.expm1(res.Vehicles)

res['Error'] =abs(res.Ratio-1)

res['Weight'] = np.expm1(res.Vehicles)/res.Prediction

res.head()
#calculates best weight

pred1  = model.predict(x_val1)

print("weight correction")

W=[(0.990+(i/1000)) for i in range(20)]

S =[]

for w in W:

    error = sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred1*w)))

    print('RMSE for {:.3f}:{:.6f}'.format(w,error))

    S.append(error)

Score = pd.Series(S,index=W)

Score.plot()

BS = Score[Score.values == Score.values.min()]

print ('Best weight for Score:{}'.format(BS))
pred=model.predict(x_val1)*1.009

np.sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred)))
x=train.drop(columns={'ID','Vehicles'},axis=1)

y=train.loc[:,['Vehicles']]

test=test.drop(columns={'ID','Vehicles'},axis=1)
model = XGBRegressor(

    max_depth=8,

    n_estimators=220,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,

    

    seed=42)



model.fit(x, y)

pred=model.predict(test)*1.009

sub['Vehicles']=np.expm1(pred)

sub.to_csv('finalsub.csv',index=False)