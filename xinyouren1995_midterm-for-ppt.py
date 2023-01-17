import numpy as np               

import pandas as pd              

import seaborn as sns             

from scipy import stats          

import calendar

from datetime import datetime    

import matplotlib.pyplot as plt  

%matplotlib inline  
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
sns.pairplot(train,x_vars=["holiday","workingday","weather","season"],

                   y_vars=["casual","registered","count"], 

             plot_kws={'alpha': 0.2})

sns.pairplot(train,x_vars=["temp","atemp","humidity","windspeed"],

                   y_vars=["casual","registered","count"], 

             plot_kws={'alpha': 0.2})

fig,axes = plt.subplots(2,2)

fig.set_size_inches(12,10)



sns.distplot(train['temp'],ax=axes[0,0])

sns.distplot(train['atemp'],ax=axes[0,1])

sns.distplot(train['humidity'],ax=axes[1,0])

sns.distplot(train['windspeed'],ax=axes[1,1])



axes[0,0].set(xlabel='temp',title='Distribtion of temp')

axes[0,1].set(xlabel='atemp',title='Distribtion of atemp')

axes[1,0].set(xlabel='humidity',title='Distribtion of humidity')

axes[1,1].set(xlabel='windspeed',title='Distribtion of windspeed')

numeric=["temp","atemp","humidity","windspeed",

         "casual","registered","count"]

corrmask=train[numeric].corr()

mask=np.array(corrmask)

mask[np.tril_indices_from(mask)]=False

fig,ax=plt.subplots()

fig.set_size_inches(20,10)



sns.heatmap(corrmask,mask=mask,vmax=1,annot=True)

temp_rentals=train.groupby(["temp"],as_index=True).agg({"casual":"mean",

                                                        "registered":"mean",

                                                        "count":"mean"})



temp_rentals.plot(title="Average rentals at different temperatures")

temp_rentals=train.groupby(["atemp"],as_index=True).agg({"casual":"mean",

                                                         "registered":"mean",

                                                         "count":"mean"})



temp_rentals.plot(title="Average rentals at different apparent temperatures")

humidity_rentals=train.groupby(["humidity"],as_index=True).agg({"casual":"mean",

                                                                "registered":"mean",

                                                                "count":"mean"})



humidity_rentals.plot(title="Average rentals at different humidities")

windspeed_rentals=train.groupby(["windspeed"],as_index=True).agg({"casual":"mean",

                                                                "registered":"mean",

                                                                "count":"mean"})

windspeed_rentals.plot(title="Average rentals in different windspeed")

train['datetime'] = pd.to_datetime(train['datetime'],errors='coerce')

train['date'] = train['datetime'].apply(lambda x: x.date())

train['year'] = train['datetime'].apply(lambda x: x.year)

train['month'] = train['datetime'].apply(lambda x: x.month)

train['weekday'] = train['datetime'].apply(lambda x: x.weekday())

train['hour'] = train['datetime'].apply(lambda x: x.hour).astype('int')
workingday_df=train[train['workingday']==1]

workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                                    'registered':'mean',

                                                                    'count':'mean'})

workingday_df.plot(title="dailyrentals on workingday")

nworkingday_df=train[train['workingday']==0]

nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                                      'registered':'mean',

                                                                      'count':'mean'})

nworkingday_df.plot(title="dailyrentals on nworkingday")

holiday_df=train[train['holiday']==1]

holiday_df = holiday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                              'registered':'mean',

                                                              'count':'mean'})



holiday_df.plot(title="dailyrentals on holiday")

nholiday_df=train[train['holiday']==0]

nholiday_df = nholiday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                                'registered':'mean',

                                                                'count':'mean'})

nholiday_df.plot(title="dailyrentals on not holiday")

plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=train,x="month",hue="year",y="casual")

plt.subplot(122)

sns.boxplot(data=train,x="month",hue="year",y="registered")

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=train,x='season',hue='year',y='casual')

plt.subplot(122)

sns.boxplot(data=train,x='season',hue='year',y='registered')

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=train,x='weather',hue='year',y='casual')

plt.subplot(122)

sns.boxplot(data=train,x='weather',hue='year',y='registered')

plt.show()
dummies_month = pd.get_dummies(train['month'], prefix= 'month')

dummies_season = pd.get_dummies(train['season'], prefix= 'season')

dummies_weather = pd.get_dummies(train['weather'], prefix= 'weather')

dummies_year = pd.get_dummies(train['year'], prefix= 'year')



data=pd.concat([train,dummies_month,dummies_season,dummies_weather,dummies_year],axis=1)



yLabels=data['count']

dropFeatures = ['casual' , 'count' , 'datetime' , 

                'registered' , 'date' ,'season',

                'weather','month','year']



dataTrain = data.drop(dropFeatures,axis=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from math import sqrt

import statsmodels.api as sm
X_train_ca, X_valid_ca, y_train_ca, y_valid_ca = train_test_split(dataTrain, 

                                    train['casual'], test_size=0.3, random_state=42)

X_train_re, X_valid_re, y_train_re, y_valid_re = train_test_split(dataTrain, 

                                    train['registered'], test_size=0.3, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(dataTrain, 

                                    train['count'], test_size=0.3, random_state=42)
rfModel_ca = RandomForestRegressor(n_estimators=1000 , 

                                   oob_score=True, random_state = 42)

rfModel_ca.fit(X_train_ca , y_train_ca)



rfModel_re = RandomForestRegressor(n_estimators=1000 , 

                                   oob_score=True, random_state = 42)

rfModel_re.fit(X_train_re , y_train_re)



rfModel = RandomForestRegressor(n_estimators=1000 , 

                                oob_score=True, random_state = 42)

rfModel.fit(X_train , y_train)
preds_train_ca = rfModel_ca.predict( X = X_train_ca)

rmsle_casual_train = sqrt(mean_squared_log_error(y_train_ca, preds_train_ca))



preds_valid_ca = rfModel_ca.predict( X = X_valid_ca)

rmsle_casual_valid = sqrt(mean_squared_log_error(y_valid_ca, preds_valid_ca))

print('Casual train rmsle : %.5f, valid rmsle : %.5f' %(rmsle_casual_train,

                                                        rmsle_casual_valid))



preds_train_re = rfModel_re.predict( X = X_train_re)

rmsle_registered_train = sqrt(mean_squared_log_error(y_train_re, preds_train_re))



preds_valid_re = rfModel_re.predict( X = X_valid_re)

rmsle_registered_valid = sqrt(mean_squared_log_error(y_valid_re, preds_valid_re))

print('Registered train rmsle : %.5f,valid rmsle : %.5f' %(rmsle_registered_train, 

                                                           rmsle_registered_valid))
preds_train = rfModel.predict( X = X_train)

rmsle_count_train = sqrt(mean_squared_log_error(y_train, preds_train))



preds_valid = rfModel.predict( X = X_valid)

rmsle_count_valid = sqrt(mean_squared_log_error(y_valid, preds_valid))

print('Count train rmsle : %.5f, valid rmsle : %.5f'  %(rmsle_count_train,

                                                        rmsle_count_valid))
preds_train_merge_count = preds_train_re + preds_train_ca

preds_valid_merge_count = preds_valid_re + preds_valid_ca

rmsle_merge_train = sqrt(mean_squared_log_error(y_train,preds_train_merge_count))

rmsle_merge_valid = sqrt(mean_squared_log_error(y_valid,preds_valid_merge_count))



print('(merge) Count train rmsle : %.5f, valid rmsle : %.5f'%(rmsle_merge_train,

                                                              rmsle_merge_valid))
rf_dict={'casual':rfModel_ca.feature_importances_.round(3),

         'registered':rfModel_re.feature_importances_.round(3),

         'count':rfModel.feature_importances_.round(3)}

pd.DataFrame(rf_dict,X_train_ca.columns)
test['datetime'] = pd.to_datetime(test['datetime'],errors='coerce')

test['date'] = test['datetime'].apply(lambda x: x.date())

test['year'] = test['datetime'].apply(lambda x: x.year)

test['month'] = test['datetime'].apply(lambda x: x.month)

test['weekday'] = test['datetime'].apply(lambda x: x.weekday())

test['hour'] = test['datetime'].apply(lambda x: x.hour).astype('int')
dummies_month = pd.get_dummies(test['month'], prefix= 'month')

dummies_season = pd.get_dummies(test['season'], prefix= 'season')

dummies_weather = pd.get_dummies(test['weather'], prefix= 'weather')

dummies_year = pd.get_dummies(test['year'], prefix= 'year')



data_test=pd.concat([test,dummies_month,dummies_season,dummies_weather,dummies_year],axis=1)

datetimecol = test['datetime']

yLabels=data['count']

dropFeatures = ['datetime' , 'date' ,'season',

                'weather','month','year']



dataTest = data_test.drop(dropFeatures,axis=1)
predsTest= rfModel_re.predict(X = dataTest) + rfModel_ca.predict(X = dataTest)

submission=pd.DataFrame({'datetime':datetimecol , 'count':[max(0,x) for x in predsTest]})
submission.to_csv('sampleSubmission.csv',index=False)
submission