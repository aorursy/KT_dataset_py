import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import preprocessing

from sklearn import metrics

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from xgboost import plot_importance, plot_tree

import xgboost as xgb
# src key = inverter id

df_gendata = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_gendata
# PLANT_ID is a unique id, so i removed it

df_gendata = df_gendata.drop(['PLANT_ID'], axis=1)

df_gendata
# re-format date_time astype

df_gendata['DATE_TIME']= pd.to_datetime(df_gendata['DATE_TIME'],format='%d-%m-%Y %H:%M') 

df_gendata
## separate date and time

df_gendata['TIME'] = df_gendata['DATE_TIME'].dt.time

df_gendata['DATE'] = pd.to_datetime(df_gendata['DATE_TIME'].dt.date)

df_gendata
# obviously, dc_power generated massively at daytime (sunlight)

df_gendata.plot(x='TIME', y='DC_POWER', style='.', figsize = (12, 8))

df_gendata.groupby('TIME')['DC_POWER'].agg('mean').plot(legend=True, colormap='rocket')

plt.ylabel('DC Power Generated')

plt.title('DC POWER plot per time')

plt.show()
# NOTE

# the data in Source_key is inconsistent 

# it had started with 21 inverter id 

# and somehow the other inverters ids started 

# For example, YxYtjZvoooNbGkE didn't start at the beginning 

# but appeared at 15-05-2020 01:00 

# Counts are not the same for each inverter

df_gendata.groupby('SOURCE_KEY').count().reset_index()
# same for AC power

df_gendata.plot(x='TIME', y='AC_POWER', style='.', figsize = (12, 8))

df_gendata.groupby('TIME')['AC_POWER'].agg('mean').plot(legend=True, colormap='rocket')

plt.ylabel('AC Power Generated')

plt.title('AC POWER plot per time')

plt.show()
# total yields are different based on inverters as well

# however, all are going up (produced more)

fig = plt.figure(figsize=(12,8))



for i in df_gendata.groupby('SOURCE_KEY').count().reset_index()['SOURCE_KEY']:

    sns.lineplot(data=df_gendata[df_gendata['SOURCE_KEY']==i], x='DATE_TIME', y='TOTAL_YIELD')

    

plt.legend(title='SOURCE_KEY',labels=df_gendata.groupby('SOURCE_KEY').count().reset_index()['SOURCE_KEY'],

           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



fig.autofmt_xdate()
# mean inverters SOURCE_KEY also vary

# this indicates a good sign for the predictive 

# as some inverters are not present all the time

fig = plt.figure(figsize=(16,10))

df_gendata.groupby('SOURCE_KEY').mean().reset_index()

sns.barplot(data=df_gendata.groupby('SOURCE_KEY').mean().reset_index(), x='SOURCE_KEY', y='DAILY_YIELD')

plt.ylim([3000,3500])

fig.autofmt_xdate()
# Sum of daily yield

# it varies from factors to factors (time, weather)

# therefore, it's better to predict daily yield than total yield

df_gendata.groupby('DATE')['DAILY_YIELD'].agg('sum').plot.bar(figsize=(10,6), legend=True)

plt.title('Daily yield')

plt.ylabel('Sum of Daily yield')
# clearly, daily yield is the accumalative of the day and restart for the next day

# thus, current_daily_yield - previous_daily_yield = yield generated for 15 minutes interval

df_gendata.plot(x='TIME', y='DAILY_YIELD', style='b.', figsize=(12,8))

df_gendata.groupby('TIME')['DAILY_YIELD'].agg('mean').plot(legend=True)

plt.title('DAILY YIELD')

plt.ylabel('Yield')

plt.show()
# make date_time string

df_gendata['DATE_TIME'] = df_gendata['DATE_TIME'].astype(str)
# double check df info

df_gendata.info()
# read the weather sensors

df_weather_sensor = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_weather_sensor['DATE_TIME'] = pd.to_datetime(df_weather_sensor['DATE_TIME'], errors='coerce')

df_weather_sensor['TIME'] = df_weather_sensor['DATE_TIME'].dt.time

df_weather_sensor['DATE'] = pd.to_datetime(df_weather_sensor['DATE_TIME'].dt.date)

df_weather_sensor 
# AMBIENT_TEMPERATURE is at its peak around afternoon

df_weather_sensor.plot(x='TIME', y='AMBIENT_TEMPERATURE', style='.', figsize = (12, 8))

df_weather_sensor.groupby('TIME')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='rocket')

plt.ylabel('AMBIENT_TEMPERATURE')

plt.title('AMBIENT_TEMPERATURE per time')

plt.show()
# MODULE_TEMPERATURE at its peak

fig = plt.figure(figsize=(10,6))

pd.plotting.register_matplotlib_converters()

sns.lineplot(data=df_weather_sensor.groupby('TIME').mean().reset_index(), x='TIME', y='MODULE_TEMPERATURE')
# AMBIENT_TEMP at its peak

fig = plt.figure(figsize=(10,6))

pd.plotting.register_matplotlib_converters()

sns.lineplot(data=df_weather_sensor.groupby('TIME').mean().reset_index(), x='TIME', y='AMBIENT_TEMPERATURE')
# IRRADIATION at its peak

fig = plt.figure(figsize=(10,6))

pd.plotting.register_matplotlib_converters()

sns.lineplot(data=df_weather_sensor.groupby('TIME').mean().reset_index(), x='TIME', y='IRRADIATION')
# all good

df_weather_sensor.info()
# PLANT_ID and SOURCE_KEY are a unique id, so i removed them

df_weather_sensor = df_weather_sensor.drop(['PLANT_ID','SOURCE_KEY','DATE','TIME'], axis=1)

df_weather_sensor
# make date_time string

df_weather_sensor['DATE_TIME'] = df_weather_sensor['DATE_TIME'].astype(str)
# merge weather sensors with generation data as they are good predictive factors

# using inner by 'DATE_TIME'

df = pd.merge(df_gendata, df_weather_sensor, on='DATE_TIME', how='inner')

df
# DC_POWER, AC_POWER, AMBIENT_TEMP, MODULE_TEMP, IRRADIATION are highly correlated

df.corr()
fig = plt.figure(figsize=(8,6))

sns.heatmap(df.corr(),cmap='coolwarm')
# performed one-hot encoder or pd.get_dummies

inverter_cap = pd.get_dummies(df['SOURCE_KEY'])

inverter_cap
# concat to the main df

df = pd.concat([df,inverter_cap],axis=1)

df
df = df.groupby('DATE').mean().reset_index()

df['DATE'] = df['DATE'].astype(str)

df.head()
# Comments in Thai



# จากที่ อ.ธนารักษ์ ท่านได้กล่าวไว้  

# โจทย์ ข้อหนึ่ง นั้น อาจมีแนวคิดหลายแนว

# ประเด็นสำคัญ คือ ถ้า เรามีข้อมูลถึงวันนี้ 

# แล้ว เรา จะสร้างโมเดลที่จะ 

# ทำนายข้อมูล Yield ในอีก 3 วันข้างหน้า

# หรือว่า อีก 7 วันข้างหน้า จะทำได้อย่างไร



# ดังนั้น ผมจึงคิดว่าการ Predict แบบ Timestep > ...,(n-1),(n) ทำนาย (n+3) , (n+7) จะได้ผลดีกว่า และได้โมเดลที่มีความแม่นยำมากกว่าการทำ Cross Validation

# โดยให้ความสำคัญของ daily yield ที่เราจะทำการ predict มากที่สุด ดังนั้นการหาค่าเฉลี่ยของปัจจัยต่างๆรวมทั้งวัน จะสามารถเป็น Features ที่มี

# ประโยชน์ต่อการทำนายค่า daily yield ในอนาคตได้ และวัดผลกับ daily yield ของจริง เพื่อเปรียบเทียบประสิทธิภาพ แสดงให้เห็นว่าเมื่อมีข้อมูลมากขึ้น Model จะมีการเรียนรู้ได้ดีขึ้น
## Predict the next 3 days

## use the first 5 days as the training 

## and predict the 8th day

## use the first 6 days as the training

## then predict the 9th day

## and so on, until the last day is predicted

## so totally 27 days are predicted from 8th to 34th

def predict_next_3days(df, model):

    y_pred = []

    X = df.drop(['DAILY_YIELD','TOTAL_YIELD','DATE'], axis=1)

    y = df['DAILY_YIELD']

    for i in range(len(df)-7):

        model.fit(X[0:i+5], y[0:i+5])

        y_pred.append(model.predict(X[i+7:i+8])[0])

    

    y_test = y.tail(len(df)-7)

    rmse = mean_squared_error(y_test,y_pred,squared=False)

    return rmse, y_pred, y_test
reg = xgb.XGBRegressor(n_estimators=600,

                       objective='reg:squarederror',

                       learning_rate=0.05,

                       colsample_bytree=0.6,

                       max_depth=16,

                       min_child_weight=2)
xgb_rmse, y_pred, y_test = predict_next_3days(df, reg)
print(xgb_rmse)

print(y_pred)
df_pred_3days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(27).to_list())



df_pred_3days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 3 days by XGBOOST')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
rf_regressor = RandomForestRegressor(max_depth=None, random_state=99)

rf_rmse, y_pred, y_test = predict_next_3days(df, rf_regressor)
print(rf_rmse)

print(y_pred)
df_pred_3days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(27).to_list())



df_pred_3days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 3 days by RANDOM FOREST')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
cb = CatBoostRegressor(iterations=100,

                       learning_rate=0.16,

                       depth=8)

cb_rmse, y_pred, y_test = predict_next_3days(df, cb)
print(cb_rmse)

print(y_pred)
df_pred_3days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(27).to_list())



df_pred_3days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_3days.index, y=df_pred_3days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 3 days by CATBOOST')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
model_lst = ['XGBoost', 'Random Forest Regressor', 'CATBoost']

model_performace = [xgb_rmse, rf_rmse, cb_rmse]

fig = plt.figure(figsize=(10,6))

sns.barplot(x=model_lst, y=model_performace)

plt.title('Model Performance Comparison')

plt.xlabel('Model Title')

plt.ylabel('Root Mean Square Error (RMSE)')

print('RMSE of XGBoost: ',xgb_rmse)

print('RMSE of Random Forest Regressor: ',rf_rmse)

print('RMSE of CATBoost: ',cb_rmse)
## Predict the next 7 days

## use the first 5 days as the training 

## and predict the 12th day

## use the first 6 days as the training

## then predict the 13th day

## and so on, until the last day is predicted

## so totally 23 days are predicted from 12nd to 34th

def predict_next_7days(df, model):

    y_pred = []

    X = df.drop(['DAILY_YIELD','TOTAL_YIELD','DATE'], axis=1)

    y = df['DAILY_YIELD']

    for i in range(len(df)-11):

        model.fit(X[0:i+5], y[0:i+5])

        y_pred.append(model.predict(X[i+11:i+12])[0])

    

    y_test = y.tail(len(df)-11)

    rmse = mean_squared_error(y_test,y_pred,squared=False)

    return rmse, y_pred, y_test
reg = xgb.XGBRegressor(n_estimators=600,

                       objective='reg:squarederror',

                       learning_rate=0.05,

                       colsample_bytree=0.6,

                       max_depth=16,

                       min_child_weight=2)
xgb_rmse, y_pred, y_test = predict_next_7days(df, reg)
print(xgb_rmse)

print(y_pred)
df_pred_7days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(23).to_list())



df_pred_7days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 3 days by XGBoost')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
rf_regressor = RandomForestRegressor(max_depth=None, random_state=99)

rmse, y_pred, y_test = predict_next_7days(df, rf_regressor)
print(rf_rmse)

print(y_pred)
df_pred_7days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(23).to_list())



df_pred_7days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 3 days by RANDOM FOREST')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
cb = CatBoostRegressor(iterations=100,

                       learning_rate=0.16,

                       depth=8)

cb_rmse, y_pred, y_test = predict_next_7days(df, cb)
print(cb_rmse)

print(y_pred)
df_pred_7days = pd.DataFrame({'daily_yield': y_test.to_list(),

                              'daily_yield_pred': y_pred}, 

                              index=df['DATE'].tail(23).to_list())



df_pred_7days
fig = plt.figure(figsize=(16,10))

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield'], marker='o')

sns.lineplot(x=df_pred_7days.index, y=df_pred_7days['daily_yield_pred'], marker='o')

plt.legend(title='Legend',labels=['daily_yield','daily_yield_pred'])

plt.title(label='daily_yield VS daily_yield_pred for the next 7 days by CATBOOST')

plt.xlabel('DATE')

plt.ylabel('Yield')

fig.autofmt_xdate()
model_lst = ['XGBoost', 'Random Forest Regressor', 'CATBoost']

model_performace = [xgb_rmse, rf_rmse, cb_rmse]

fig = plt.figure(figsize=(10,6))

sns.barplot(x=model_lst, y=model_performace)

plt.title('Model Performance Comparison')

plt.xlabel('Model Title')

plt.ylabel('Root Mean Square Error (RMSE)')

print('RMSE of XGBoost: ',xgb_rmse)

print('RMSE of Random Forest Regressor: ',rf_rmse)

print('RMSE of CATBoost: ',cb_rmse)