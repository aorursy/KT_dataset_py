import pandas as pd
import os
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.graph_objects as go
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
os.chdir("/kaggle/input/nyc-taxi")
df = pd.read_csv('train.csv')
df1 = df
df.head()
df.info()
df.isnull().sum()
df.describe()
df.corr()
#onehot encoding for vendor_id
df['vendor_id'] = df['vendor_id'].replace(2,0)
df.columns
df['passenger_count'].value_counts()
#Droping rows belonging to passenger no 7,8,9, as they are rare hence no learning and also droping 0 passenger as they belong to idle or rest by the drivers
df = df[(df['passenger_count'] != 0) & (df['passenger_count'] != 7) & (df['passenger_count'] != 8) & (df['passenger_count'] != 9)]
df['store_and_fwd_flag'].value_counts()
#we are also droping 'store_and_fwd_flag' column as it does not have any influence on out come
from math import radians, cos, sin, asin, sqrt 
def distance(lat1, lat2, lon1, lon2): 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers.
    r = 6371
       
    # calculate the result 
    return(c * r) 

la1 = df1['pickup_latitude']
la2 = df1['dropoff_latitude']
lo1 = df1['pickup_longitude']
lo2 = df1['dropoff_longitude']
dis = []
for i in zip(la1,la2,lo1,lo2) :
    dis.append(distance(i[0],i[1],i[2],i[3]))
df['distance'] = pd.Series(dis)
df['distance'] 
df['speed'] = (df['distance']/df['trip_duration']) * 3600
df.head()
#ProfileReport(df1)
sns.distplot(df['trip_duration'])
np.sum((df['trip_duration'] < 3000) & (df['trip_duration'] > 120)) 
df = df[(df['trip_duration'] < 3000) & (df['trip_duration'] > 120)]
sns.distplot(df['trip_duration'])

#Normalising data
mmc = MinMaxScaler()
X_scale = mmc.fit_transform(df[['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'distance']])
df[['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'distance']] = X_scale
#converting in to date format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
hour = df['pickup_datetime'].apply(lambda x : x.hour)
df_hour = df
df_hour['hour'] = hour 
df_hour.head()
np.sort(df_hour.groupby('hour').mean()['speed'])
def classify(x):
    if x < 13 :
        return 1
    elif x < 18:
        return 2
    elif x < 20:
        return 3
    elif x < 25:
        return 4
avg_speed_list = list(df_hour.groupby('hour').mean()['speed'])
df_hour['avg_speed_by_hour'] = df_hour['hour'].apply(lambda x : avg_speed_list[x])
df_hour['bin'] =  df_hour['avg_speed_by_hour'].apply(classify)
df_hour['bin']
'''
#onehot encoding for hour,day,and month 
month = df['pickup_datetime'].apply(lambda x : int(x.month))
month = pd.get_dummies(month, prefix= 'month', drop_first=True)

hour = df['pickup_datetime'].apply(lambda x : x.hour)
hour = pd.get_dummies(hour, prefix= 'hour', drop_first=True)

day = df['pickup_datetime'].apply(lambda x : x.day)
day = pd.get_dummies(day, prefix= 'day', drop_first=True)
'''

#onehot encoding for hour,day,and month 
month = df['pickup_datetime'].apply(lambda x : int(x.month))
month = pd.get_dummies(month, prefix= 'month', drop_first=True)

hour = pd.get_dummies(df_hour['bin'], prefix= 'hour_bin', drop_first=True)

day = df['pickup_datetime'].apply(lambda x : x.day)
day = pd.get_dummies(day, prefix= 'day', drop_first=True)

df_f = pd.concat([df,month,hour,day],1)
df_f
X = df[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'distance']]
X = pd.concat([X,month,hour,day],1)
y = df['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train,y_train)
print(model)
y_pred = model.predict(data=X_test)
final_df = pd.DataFrame()
final_df["Prediction"] = y_pred
final_df["Actual"] = y_test.values
final_df
test_mse = mean_squared_error(y_true=final_df["Actual"], y_pred=final_df["Prediction"])
test_mae = mean_absolute_error(y_true=final_df["Actual"], y_pred=final_df["Prediction"])
print('Test MAE: {}'.format(test_mae))
print('Test MSE: {}'.format(test_mse))
rmse = np.sqrt(test_mse) / 60
rmse
y_avg = np.array([y_test.mean()] * len(y_test))
null_mse = mean_squared_error(y_true=final_df["Actual"], y_pred=y_avg)
null_mae = mean_absolute_error(y_true=final_df["Actual"], y_pred=y_avg)
print('Null MAE: {}'.format(null_mae))
print('Null MSE: {}'.format(null_mse))
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(y=final_df['Prediction'].values,
                    mode='lines',
                    name='Prediction'))
fig.add_trace(go.Scatter(y=final_df['Actual'].values,
                    mode='lines',
                    name='Actual'))

fig.update_xaxes(rangeslider_visible=True)
fig.show()
