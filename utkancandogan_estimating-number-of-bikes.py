# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import seaborn as sns

sns.set(font_scale=1.5)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output



print(check_output(["ls", "../input/"]).decode("utf8"))



    

    

    

df_train= pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_joined = df_train.append(df_test,ignore_index=True)



print(df_joined.head(100))

print(df_test.head(100))



print(df_train.describe())

# we will need this error function later.

def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
# Let's engineer the data

df_joined['Month'] = df_joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)

df_joined['Hour'] = df_joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

df_joined['Weekday'] = df_joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())



#df_joined['Weekday'] = df_joined['Weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})

#df_joined['season'] = df_joined['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})

#df_joined['weather'] = df_joined['weather'].map({1:'Clear',2:'Mist',3:'Light Rain',4:'Heavy Rain'})



datetimecol = df_test["datetime"]



df_joined.drop('datetime',inplace=True,axis=1)

df_train = df_joined[ df_joined['count'].notnull()]

df_test = df_joined[ df_joined['count'].isnull()]

# Let's add some visualizations, inspired by those of Vivek Srinivasan. Thanks Vivek!



fig,axes= plt.subplots(2,2,figsize=(20,20))

sns.boxplot(data=df_train, y='count',x='weather',ax=axes[0,0])

sns.boxplot(data=df_train, y='count',x='season',ax=axes[0,1])

sns.boxplot(data=df_train, y='count',x='Weekday',ax=axes[1,0])

sns.boxplot(data=df_train, y='count',x='Month',ax=axes[1,1])



df_Month_grouped = df_train[['count','Month']].groupby('Month',as_index=False).mean()

df_Hour_season_grouped = (df_train.groupby(["Hour","season"],as_index=False)["count"].mean())

df_Hour_Weekday_grouped = (df_train.groupby(["Hour","Weekday"],as_index=False)["count"].mean())

df_Hour_Month_grouped = (df_train.groupby(["Hour","Month"],as_index=False)["count"].mean())

df_Hour_weather_grouped = (df_train.groupby(["Hour","weather"],as_index=False)["count"].mean())



fig,axes= plt.subplots(5,1,figsize=(30,30))

sns.barplot( x = df_Month_grouped['Month'], y = df_Month_grouped['count'], ax=axes[0])

sns.pointplot( y = df_Hour_season_grouped['count'], x = df_Hour_season_grouped['Hour'],  hue=df_Hour_season_grouped['season'], join=True,ax=axes[1])

sns.pointplot( y = df_Hour_Weekday_grouped['count'], x = df_Hour_Weekday_grouped['Hour'],  hue=df_Hour_Weekday_grouped['Weekday'], join=True,ax=axes[2])

sns.pointplot( y = df_Hour_Month_grouped['count'], x = df_Hour_Month_grouped['Hour'],  hue=df_Hour_Month_grouped['Month'], join=True,ax=axes[3])

sns.pointplot( y = df_Hour_weather_grouped['count'], x = df_Hour_weather_grouped['Hour'],  hue=df_Hour_weather_grouped['weather'], join=True,ax=axes[4])





#sns.countplot(data=df_train, x='Hour', hue='season',ax=axes[0,1])

sns.heatmap(data=df_train.corr(), linewidths=2)
df_test.drop(['casual','registered','count','atemp'],inplace=True,axis=1)

df_train.drop(['casual','registered','atemp'],inplace=True,axis=1)

from sklearn.model_selection import train_test_split

X_train,X_cv,y_train,y_cv = train_test_split(df_train.drop('count',axis=1),df_train['count'],test_size=0.2)
print(df_train.columns)
from sklearn.ensemble import RandomForestRegressor



RFR = RandomForestRegressor(n_estimators = 200, max_depth=10 )

RFR.fit(X_train, np.log1p(y_train))

preds = RFR.predict(X = X_cv)

print('Random forest rmsle =', rmsle(np.exp([max(x,0) for x in preds]),y_cv,False))

from sklearn.ensemble import GradientBoostingRegressor

GBM = GradientBoostingRegressor(n_estimators=2000)

GBM.fit(X_train, np.log1p(y_train))

preds = GBM.predict(X = X_cv)

print('GBM rmsle =', rmsle( np.exp([max(x,0) for x in preds]) ,y_cv,False))
finalEstimator = GradientBoostingRegressor(n_estimators=2000)

GBM.fit(df_train.drop('count',axis=1),np.log1p(df_train['count']))

preds = GBM.predict(df_test)





submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(preds)]

    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)