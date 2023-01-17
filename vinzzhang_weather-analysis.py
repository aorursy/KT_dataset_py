import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
df = pd.read_csv('../input/weather_data.csv')
cols = df.loc[15].values
df.drop(df.index[:16],inplace=True)
df.columns = cols
df = df.reset_index(drop=True)

df.head()




df.isnull().sum().sort_values(ascending = False).head(21)
df.dropna(axis=1, how='all',inplace=True)
df.drop(['Hmdx','Wind Chill','Data Quality'],axis=1, inplace=True)
df = df[df['Weather'].notnull()]
df=df.reset_index(drop=True)
df.isnull().sum().sort_values(ascending = False).head()
df[df['Wind Dir (10s deg)'].isnull()]
df['Time'] = df['Time'].str.replace(':00','')
#Convert to numeric type, except 'Date/Time' and 'Weather'
to_be_converted_cols = df.columns[1:-1] 
df[to_be_converted_cols] = df[to_be_converted_cols].apply(pd.to_numeric)
#Imputate missing variable by medians under each relative weather condition
df[df['Weather'] == 'Mostly Cloudy'] = df[df['Weather'] == 'Mostly Cloudy'].fillna(df[df['Weather'] == 'Mostly Cloudy'].median())
df[df['Weather'] == 'Mainly Clear'] = df[df['Weather'] == 'Mainly Clear'].fillna(df[df['Weather'] == 'Mainly Clear'].median())
df[df['Weather'] == 'Clear'] = df[df['Weather'] == 'Clear'].fillna(df[df['Weather'] == 'Clear'].median())
df[df['Weather'] == 'Cloudy'] = df[df['Weather'] == 'Cloudy'].fillna(df[df['Weather'] == 'Cloudy'].median())
# Missing data check
df.isnull().any().any()
df.groupby('Weather').size().sort_values(ascending = False)
# df
#Cloudy merge:
df.loc[df['Weather'] == 'Mostly Cloudy', 'Weather'] = 'Cloudy' # df[df['Weather'] == 'Mostly Cloudy']['Weather'] = 'Cloudy' //Copy Error

#Clear merge:
df.loc[df['Weather'] == 'Mostly Clear', 'Weather'] = 'Clear'
df.loc[df['Weather'] == 'Mainly Clear', 'Weather'] = 'Clear'

#Rain merge:
df.loc[df['Weather'] == 'Rain Showers', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Moderate Rain', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Drizzle', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Heavy Rain', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Thunderstorms', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Moderate Rain Showers', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Moderate Rain,Drizzle', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Thunderstorms,Rain Showers', 'Weather'] = 'Rain'
df.loc[df['Weather'] == 'Rain,Drizzle', 'Weather'] = 'Rain'

#Snow merge:
df.loc[df['Weather'] == 'Snow Showers', 'Weather'] = 'Snow'
df.loc[df['Weather'] == 'Moderate Snow', 'Weather'] = 'Snow'
df.loc[df['Weather'] == 'Ice Pellets', 'Weather'] = 'Snow'


#Fog merge:
df.loc[df['Weather'] == 'Freezing Fog', 'Weather'] = 'Fog'

#Rain & Fog merge:
df.loc[df['Weather'] == 'Moderate Rain,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Drizzle,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Rain,Drizzle,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Rain Showers,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Heavy Rain,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Freezing Rain,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Moderate Rain Showers,Fog', 'Weather'] = 'Rain,Fog'
df.loc[df['Weather'] == 'Moderate Rain,Fog', 'Weather'] = 'Rain,Fog'

#Rain & Snow merge:
df.loc[df['Weather'] == 'Rain Showers,Snow Showers', 'Weather'] = 'Rain,Snow'
df.loc[df['Weather'] == 'Rain Showers,Snow Pellets', 'Weather'] = 'Rain,Snow'
df.loc[df['Weather'] == 'Rain,Ice Pellets', 'Weather'] = 'Rain,Snow'


#Snow & Fog merge:
df.loc[df['Weather'] == 'Rain Showers,Snow Showers', 'Weather'] = 'Snow,Fog'
df.loc[df['Weather'] == 'Snow,Ice Pellets,Fog', 'Weather'] = 'Snow,Fog'
df.loc[df['Weather'] == 'Moderate Snow,Fog', 'Weather'] = 'Snow,Fog'
df.loc[df['Weather'] == 'Rain Showers,Snow Showers', 'Weather'] = 'Snow,Fog'
df.loc[df['Weather'] == 'Rain Showers,Snow Showers', 'Weather'] = 'Snow,Fog'

#Rain, Snow & Fog merge:
df.loc[df['Weather'] == 'Rain Showers,Snow Showers,Fog', 'Weather'] = 'Rain,Snow,Fog'
df.loc[df['Weather'] == 'Heavy Rain,Moderate Hail,Fog', 'Weather'] = 'Rain,Snow,Fog'
df.loc[df['Weather'] == 'Heavy Rain Showers,Moderate Snow Pellets,Fog', 'Weather'] = 'Rain,Snow,Fog'

df.groupby('Weather').size().sort_values(ascending = False)
df.head()
# Classification Start
# X = df.iloc[:,1:-1]
# y = df.iloc[:,-1]
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingRegressor
# X_train, X_test, y_train, y_test = train_test_split(X,y)
# svc = GradientBoostingRegressor()
# svc.fit(X,y)
# svc.score(X,y)









