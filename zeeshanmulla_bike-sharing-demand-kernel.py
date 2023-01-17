import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True,style="ticks")
train = pd.read_csv('../input/bikesharingdemand/train.csv')
test = pd.read_csv('../input/bikesharingdemand/test.csv')
train.head(5)
print("Train shape : {}, Test Shape : {}".format(train.shape,test.shape))
fig,ax = plt.subplots(figsize=(20,4),ncols=4,nrows=1)
sns.violinplot(x="season",y="count",hue="workingday",split=True,data=train,ax=ax[0]).set_title("Demand per season")
sns.barplot(x="season",y="count",hue="workingday",dodge=True,data=train,ax=ax[1]).set_title("Count per season")
sns.countplot(x="season",hue="workingday",dodge=True,data=train,ax=ax[2]).set_title("Demand per season per workingday")
sns.countplot(x="workingday",hue="holiday",dodge=True,data=train,ax=ax[3]).set_title("Holiday vs Working Day Count")
sns.jointplot(x="temp",y="atemp",kind="hex",data=train)
fig,ax = plt.subplots(figsize=(20,4),ncols=2,nrows=2)
sns.scatterplot(x="atemp",y="count",hue="workingday",data=train,ax=ax[0][0])
sns.distplot(train["windspeed"],ax=ax[0][1])
sns.scatterplot(x="temp",y="count",hue="workingday",data=train,ax=ax[1][0])
sns.scatterplot(x="temp",y="windspeed",hue="workingday",data=train,ax=ax[1][1])
print("temp-atemp correlation : {}".format(train['temp'].corr(train['atemp'])))
print("temp-count correlation : {}".format(train['temp'].corr(train['count'])))
print("atemp-count correlation : {}".format(train['atemp'].corr(train['count'])))
print("windspeed-count correlation : {}".format(train['windspeed'].corr(train['count'])))
"""
Lets utilize datetime columns also
as it gives information about time of booking
Lets create new information giving information about current hour
"""
import datetime
def getHour(dt):
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    return int(dt.strftime('%H'))
def getMonth(dt):
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    return int(dt.strftime('%m'))
#vectorizer is used to perform operation on whole numpy array
vectorizer = np.vectorize(getHour)
train['hour'] = vectorizer(train['datetime'].values)
train['month']  = np.vectorize(getMonth)(train['datetime'].values)
train.head(3)
"""Lets create one more variable time of the day"""
def getTime(hour):
    if hour<6:
        return "midnight"
    elif hour<10:
        return "morning"
    elif hour<17:
        return "noon"
    elif hour<23:
        return "night"
    else:
        return "midnight"
vectorizer = np.vectorize(getTime)
train['time'] = vectorizer(train['hour'].values)
train.head(3)
"""Lets visualize Hour variable"""
fig,ax = plt.subplots(figsize=(20,4),ncols=3,nrows=1)

sns.scatterplot(x="hour",y="count",hue="workingday",data=train,ax=ax[0])
sns.barplot(x="time",y="count",hue="weather",data=train,ax=ax[1])
sns.barplot(x="time",y="count",hue="workingday",data=train,ax=ax[2])
print("season-month correlation : {}".format(train['season'].corr(train['month'])))
#check for null values
print("NA Values")
for col in train.columns:
    lna = len(train[train[col].isna()])
    print("{} : {}".format(col,lna))
test['hour'] = np.vectorize(getHour)(test['datetime'])
test['time'] = np.vectorize(getTime)(test['hour'])
test['month'] = np.vectorize(getMonth)(test['datetime'])
test.head(2)
train.drop(['season','registered','casual'],axis=1,inplace=True)
Y = train['count']
df = train.append(test,sort=False,ignore_index=True)
df.head(2)
df.drop(['time','season','count'],axis=1,inplace=True)
#OneHotEncoding Categorical Variables
df = pd.get_dummies(df,columns=['month','hour','weather'],prefix_sep='_')
df['humidity'] = df['humidity'].astype('float')
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
scaler = MinMaxScaler()
df['temp'] = scaler.fit_transform(df['temp'].values.reshape(-1,1)).reshape(1,-1)[0]
df['atemp'] = scaler.fit_transform(df['atemp'].values.reshape(-1,1)).reshape(1,-1)[0]
df['windspeed'] = scaler.fit_transform(df['windspeed'].values.reshape(-1,1)).reshape(1,-1)[0]
df['humidity'] = scaler.fit_transform(df['humidity'].values.reshape(-1,1)).reshape(1,-1)[0]
df.head(3)
#df.drop("atemp",axis=1,inplace=True)
#df.drop("holiday",axis=1,inplace=True) #gives less accuracy
X = df[:len(train)].drop("datetime",axis=1)
X_test_datetime = df[len(train):].datetime
X_test = df[len(train):].drop("datetime",axis=1)
#train test split 0.25
from sklearn.model_selection import train_test_split
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=0.20)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
#1. SGD Regression 
from sklearn.linear_model import SGDRegressor
sgdregressor = SGDRegressor(max_iter=1000)
sgdregressor.fit(X_train,Y_train)
mean_absolute_error(Y_validation,sgdregressor.predict(X_validation))
#2. SVR 
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train,Y_train)
mean_absolute_error(Y_validation,svr.predict(X_validation))
#XGBRegressor 50
from xgboost import XGBRegressor
param_grid = {
    'max_depth':[5,7,10],
    'n_estimators':[90,100,130],
    'learning_rate':[0.05]
}
gscv = GridSearchCV(XGBRegressor(),param_grid=param_grid)
gscv.fit(X_train,Y_train)
best_xgb = gscv.best_estimator_
print(best_xgb)
mean_absolute_error(Y_validation,best_xgb.predict(X_validation))
#LightGBM Regressor 48
from lightgbm import LGBMRegressor
param_grid = {
    'num_leaves':[35,40,50],
    'max_depth':[-1],
    'n_estimators':[160,170,180],
    'learning_rate':[0.05]
}
gscv = GridSearchCV(LGBMRegressor(n_jobs=-1),param_grid=param_grid)
gscv.fit(X_train,Y_train)
best_lgbm = gscv.best_estimator_
print(best_lgbm)
mean_absolute_error(Y_validation,best_lgbm.predict(X_validation))
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

nnmodel = Sequential()
nnmodel.add(Dense(1028,activation='relu',input_shape=(46,)))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(524,activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(128,activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(1,activation='linear'))

nnmodel.compile(loss="mean_squared_error",optimizer="adam",metrics=['mean_absolute_error'])

epochs = 20
batch_size=8

nnmodel.fit(
    x=X_train,y=Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_validation,Y_validation),
    verbose=2
)
nnmodel.fit(x=X,y=Y,batch_size=batch_size,epochs=epochs,verbose=0)
pred = nnmodel.predict(X_test)
pred = pred.reshape(1,-1)[0]
pred[:10]
result = pd.DataFrame({
    'datetime':X_test_datetime.values,
    'count':np.ceil(np.abs(pred))
})
result.to_csv('sol.csv',index=False)
