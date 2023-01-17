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
sns.boxplot(train['count'])
train_withoutliers = train[np.abs(train['count']-train['count'].mean()) <= (3*train['count'].std())]
fig,ax=plt.subplots(2,1,figsize=(10,10)) 

sns.distplot(train_withoutliers["count"],ax=ax[0])  

stats.probplot(train_withoutliers["count"],dist="norm",fit=True,plot=ax[1])

print('Skewness :{0} '.format(train_withoutliers['count'].skew()))

print('Kurt : {0}'.format(train_withoutliers['count'].kurt()))
fig,ax=plt.subplots(2,1,figsize=(10,10))



train['count_log'] = np.log1p(train['count']+1)

sns.distplot(train['count_log'],ax=ax[0])

stats.probplot(train["count_log"], dist='norm', fit=True, plot=ax[1])
data = pd.concat([train_withoutliers,test],ignore_index=True)

data.shape
data['datetime'] = pd.to_datetime(data['datetime'],errors='coerce')



data['year'] = data['datetime'].apply(lambda x: x.year)
data['hour'] = data['datetime'].apply(lambda x: x.hour).astype('int')
data['weekday'] = data['datetime'].apply(lambda x: x.weekday())



data['date'] = data['datetime'].apply(lambda x: x.date())

data[["date","weekday"]].head()



data['month'] = data['datetime'].apply(lambda x: x.month)
fig,axes = plt.subplots(2,2)

fig.set_size_inches(12,10)



sns.distplot(data['temp'],ax=axes[0,0])

sns.distplot(data['atemp'],ax=axes[0,1])

sns.distplot(data['humidity'],ax=axes[1,0])

sns.distplot(data['windspeed'],ax=axes[1,1])



axes[0,0].set(xlabel='temp',title='Distribtion of temp')

axes[0,1].set(xlabel='atemp',title='Distribtion of atemp')

axes[1,0].set(xlabel='humidity',title='Distribtion of humidity')

axes[1,1].set(xlabel='windspeed',title='Distribtion of windspeed')
numeric=["temp","atemp","casual","registered","humidity","windspeed","count"]



corrmask=data[numeric].corr()

mask=np.array(corrmask)

mask[np.tril_indices_from(mask)]=False

fig,ax=plt.subplots()

fig.set_size_inches(20,10)



sns.heatmap(corrmask,mask=mask,vmax=1,annot=True)
sns.pairplot(data,x_vars=["holiday","workingday","weather","season",

                          "weekday","year"],

                   y_vars=["casual","registered","count"], plot_kws={'alpha': 0.2})
sns.pairplot(data,x_vars=["hour","humidity","temp"],

                   y_vars=["casual","registered","count"], plot_kws={'alpha': 0.2})

workingday_df=data[data['workingday']==1]

nworkingday_df=data[data['workingday']==0]





workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                                    'registered':'mean',

                                                                    'count':'mean'})



nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',

                                                                      'registered':'mean',

                                                                      'count':'mean'})



workingday_df.plot(title="dailyrentals on workingday")    #pandas中的绘图函数



nworkingday_df.plot(title="dailyrentals on nworkingday") 
temp_rentals=data.groupby(["temp"],as_index=True).agg({"casual":"mean","registered":"mean","count":"mean"})

temp_rentals.plot(title="Average rentals at different temperatures")
humidity_rentals=data.groupby(["humidity"],as_index=True).agg({"casual":"mean","registered":"mean","count":"mean"})

humidity_rentals.plot(title="Average rentals at different humidity")
windspeed_rentals=data.groupby(["windspeed"],as_index=True).agg({"casual":"mean",

                                                                "registered":"mean",

                                                                "count":"mean"})

windspeed_rentals.plot(title="Average rentals in different windspeed")
df1=data[data['windspeed']>40]

df2=df1[df1["count"]>170]
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=data[data["count"].notnull()],x="month",hue="year",y="casual")

plt.subplot(122)

sns.boxplot(data=data[data["count"].notnull()],x="month",hue="year",y="registered")

plt.show()
data[data['count'].notnull()].groupby(['season','year'])[['casual','registered']].sum()
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=data[data['count'].notnull()],x='season',hue='year',y='casual')

plt.subplot(122)

sns.boxplot(data=data[data['count'].notnull()],x='season',hue='year',y='registered')

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(data=data[data['count'].notnull()],x='weather',hue='year',y='casual')

plt.subplot(122)

sns.boxplot(data=data[data['count'].notnull()],x='weather',hue='year',y='registered')

plt.show()
dummies_month = pd.get_dummies(data['month'], prefix= 'month')

dummies_season = pd.get_dummies(data['season'], prefix= 'season')

dummies_weather = pd.get_dummies(data['weather'], prefix= 'weather')

dummies_year = pd.get_dummies(data['year'], prefix= 'year')
data=pd.concat([data,dummies_month,dummies_season,dummies_weather,dummies_year],axis=1)

dataTrain = data[pd.notnull(data['count'])]

dataTest= data[~pd.notnull(data['count'])].sort_values(by=['datetime'])

datetimecol = dataTest['datetime']

yLabels = dataTrain['count']



yLabels_log = np.log(yLabels+1)

dropFeatures = ['casual' , 'count' , 'datetime' , 'registered' , 'date' ,

               'windspeed' , 'atemp' ,'season','weather','month','year']



dataTrain = dataTrain.drop(dropFeatures,axis=1)

dataTest = dataTest.drop(dropFeatures,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(dataTrain, 

                                    yLabels_log, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=1000 , oob_score=True, random_state = 42)



rfModel.fit(dataTrain , yLabels_log.ravel())



#preds = rfModel.predict( X = dataTrain)
predsTest= rfModel.predict(dataTest)

#predsTest_merge= np.exp(model_re.predict(dataTest))+np.exp(model_ca.predict(dataTest))



submission=pd.DataFrame({'datetime':datetimecol , 'count':[max(0,x) for x in (np.exp(predsTest)-1)]})

submission.to_csv('bike_predictions.csv',index=False)
submission
rf_dict={'count':rfModel.feature_importances_.round(3)}

pd.DataFrame(rf_dict,X_train.columns)