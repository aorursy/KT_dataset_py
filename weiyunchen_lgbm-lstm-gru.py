import warnings

warnings.filterwarnings('ignore')

!pip install plotly_express

!wget -nc https://codeload.github.com/weiyunchen/data/zip/master

!unzip master
import os

import math

import numpy as np

import pandas as pd



import datetime

from datetime import datetime

from math import sqrt

import lightgbm as lgb

from keras.models import Sequential

from keras import backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras import regularizers

from keras import layers

from keras.layers import LSTM,GRU

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from scipy import stats

from scipy.stats import norm, skew 

import plotly_express as px

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use('ggplot')



large = 18; med = 15; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

sns.set_style("white")

%matplotlib inline
train = pd.read_csv('data-master/test.csv')

test = pd.read_csv('data-master/test.csv')



train.rename(columns={'Lane 1 Flow (Veh/5 Minutes)':'flow','5 Minutes':'time' },inplace = True)

test.rename(columns={'Lane 1 Flow (Veh/5 Minutes)':'flow','5 Minutes':'time' },inplace = True)
def null_count(data):  # 定义 null 值查找函数，函数名 null_count

    null_data = data.isnull().sum()  # 查找各个特征 null 值并计算数量

    null_data = null_data.drop(null_data[null_data == 0].index).sort_values(

        ascending=False)  # 删除数目为零的特征，降序排列

    return null_data  # 返回结果



null_count(train)  # 调用 null_count 函数统计 data 的 null，输出结果
plt.figure(figsize=(25,10), dpi= 80)

plt.plot('time', 'flow', data=train, color='tab:blue')



plt.ylim(0, 200)

xtick_location = train['time'].tolist()[::1555]

plt.xticks(xtick_location,xtick_location, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)

plt.yticks(fontsize=12, alpha=.7)

plt.title("Traffic Flow(per 5 mins)", fontsize=20)

plt.grid(axis='both', alpha=.3)



plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   

plt.show()
train1=train.head(301)



train2=train1.drop(axis=1, index=0)

train2.index=train2.index-1

train2['flow_prev5min']=train1['flow'] # flow_prev5min为五分钟前的交通流量

train2['change_5min']=train2['flow']/train2['flow_prev5min']



train0=train.head(600)

train3=train0.drop(axis=1, index=range(288))

train3.index=train3.index-288

train3['flow_prev1day']=train0['flow'] # flow_prev1day为一天前的交通流量

train3['change_1day']=train3['flow']/train3['flow_prev1day']
# 通过动态时序图像观察每个时刻的当时交通流量和5分钟前的交通流量变化情况

import plotly_express as px



px.scatter(train2,x="flow_prev5min",y="flow",size="change_5min",size_max=35,

           animation_frame="time",range_x=[0,200], range_y=[0,200],width=900, height=900)
# 通过动态时序图像观察每个时刻的当时交通流量和1天前的交通流量变化情况

px.scatter(train3,x="flow_prev1day",y="flow",size="change_1day",size_max=35,

           animation_frame="time",range_x=[0,200], range_y=[0,200],width=900, height=900)
def get_hour(time):

    hour = time[-5:-3]

    return hour



def get_min(time):

    min = time[-2:]

    return min



def get_day(time):

    day = time[:2]

    return day



def get_month(time):

    month = time[3:5]

    return month



def get_year(time):

    year = time[6:10]

    return year



def get_line(time):

    line = time[2:3]

    return line



train['hour'] = train['time'].apply(get_hour)

train['min'] = train['time'].apply(get_min)

train['day'] = train['time'].apply(get_day)

train['month'] = train['time'].apply(get_month)

train['year'] = train['time'].apply(get_year)

train['line'] = train['time'].apply(get_line)



# 生成yymmdd格式的日期

train['yymmdd']=train['year']+train['line']+train['month']+train['line']+train['day']

# 生成yymmdd格式的时间

train['yymmddh']=train['year']+train['line']+train['month']+train['line']+train['day']+train['line']+train['hour']
test['hour'] = test['time'].apply(get_hour)

test['min'] = test['time'].apply(get_min)

test['day'] = test['time'].apply(get_day)

test['month'] = test['time'].apply(get_month)

test['year'] = test['time'].apply(get_year)

test['line'] = test['time'].apply(get_line)



# 生成yymmdd格式的日期

test['yymmdd']=test['year']+test['line']+test['month']+test['line']+test['day']

# 生成yymmdd格式的时间

test['yymmddh']=test['year']+test['line']+test['month']+test['line']+test['day']+test['line']+test['hour']
grb_day_flow_sum=train.groupby('yymmdd')['flow'].sum()



day_flow_sum = pd.DataFrame({"date" : grb_day_flow_sum.index, "day_flow":grb_day_flow_sum})

day_flow_sum.index = range(len(grb_day_flow_sum))





grb_hour_flow_sum=train.groupby('hour')['flow'].sum()



hour_flow_sum = pd.DataFrame({"hour" : grb_hour_flow_sum.index, "hour_flow":grb_hour_flow_sum})

hour_flow_sum.index = range(len(grb_hour_flow_sum))
plt.figure(1)

plt.figure(figsize=(10,11), dpi= 80)



plt.subplot(211)

plt.plot('date', 'day_flow', data=day_flow_sum, color='tab:blue')



plt.ylim(0, 30000)

xtick_location = day_flow_sum['date'].tolist()[::4]

plt.xticks(xtick_location,xtick_location, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)

plt.yticks(fontsize=12, alpha=.7)

plt.title("Traffic Flow(per day)", fontsize=20)

plt.grid(axis='both', alpha=.3)



plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   



plt.subplot(212)

plt.plot('hour', 'hour_flow', data=hour_flow_sum, color='tab:blue')



plt.ylim(0, 50000)

xtick_location = hour_flow_sum['hour'].tolist()[::1]

plt.xticks(xtick_location,xtick_location, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)

plt.yticks(fontsize=12, alpha=.7)

plt.title("Traffic Flow(hour)", fontsize=20)

plt.grid(axis='both', alpha=.3)



plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   

plt.show()
every_hour_flow = train.groupby('yymmddh').sum()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)

plot_acf(every_hour_flow.flow.tolist(), ax=ax1, lags=90)

plot_pacf(every_hour_flow.flow.tolist(), ax=ax2, lags=50)



ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)

ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)

ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)

ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)



ax1.tick_params(axis='both', labelsize=12)

ax2.tick_params(axis='both', labelsize=12)

plt.show()
# 对类别特征编码

lb=LabelEncoder()

train['hour_code'] = lb.fit_transform(train['hour'].values)

train['month_code'] = lb.fit_transform(train['month'].values)

train['day_code'] = lb.fit_transform(train['day'].values)

train['min_code'] = lb.fit_transform(train['min'].values)



test['hour_code'] = lb.fit_transform(test['hour'].values)

test['month_code'] = lb.fit_transform(test['month'].values)

test['day_code'] = lb.fit_transform(test['day'].values)

test['min_code'] = lb.fit_transform(test['min'].values)
col=['hour_code', 'day_code','month_code','min_code']



X=train[col]

y=train['flow']



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
from matplotlib.pylab import date2num



train['time1'] = pd.to_datetime(train['time'],format="%d/%m/%Y %H:%M")

train['timestamp'] = date2num(train['time1'])



test['time1'] = pd.to_datetime(test['time'],format="%d/%m/%Y %H:%M")

test['timestamp'] = date2num(test['time1'])
lgb_train = lgb.Dataset(X_train, y_train, feature_name=col, categorical_feature=col)

lgb_test = lgb.Dataset(X_test, y_test, feature_name=col, categorical_feature=col, reference=lgb_train)



# 设置参数

params = {'nthread': 32,  # 进程数

              'objective': 'regression',

              'learning_rate':0.001,

              #'num_leaves': 64, 

              #'max_depth': 6, 

              'feature_fraction': 0.7,  # 样本列采样

              'lambda_l1':0.001,  # L1 正则化

              'lambda_l2': 0,  # L2 正则化

              'bagging_seed': 100,  # 随机种子

              }

params['metric'] = ['rmse']



evals_result = {}  #记录训练结果



print('START LGBM')

gbm_start=datetime.now() 

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=[lgb_train, lgb_test],

                evals_result=evals_result,

                verbose_eval=10)

gbm_end=datetime.now() 

print('spendt time :'+str((gbm_end-gbm_start).seconds)+'(s)')
ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()
ax = lgb.plot_importance(gbm, max_num_features=7)

plt.show()
test_lgb=gbm.predict(test[col]) 



test['flow_LGBM']=test_lgb
train = pd.read_csv('data-master/test.csv')

test = pd.read_csv('data-master/test.csv')



train.rename(columns={'Lane 1 Flow (Veh/5 Minutes)':'flow','5 Minutes':'time' },inplace = True)

test.rename(columns={'Lane 1 Flow (Veh/5 Minutes)':'flow','5 Minutes':'time' },inplace = True)
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train0['flow'].values.reshape(-1, 1))



flow_train_now = scaler.transform(train['flow'].values.reshape(-1, 1)).reshape(1, -1)[0]

flow_test_now = scaler.transform(test['flow'].values.reshape(-1, 1)).reshape(1, -1)[0]



tr, te = [], []

for i in range(1, len(flow_train_now)):

    tr.append(flow_train_now[i - 1: i + 1])

for i in range(1, len(flow_test_now)):

    te.append(flow_test_now[i - 1: i + 1])



tr = np.array(tr)

te = np.array(te)

np.random.shuffle(tr)
X_train = tr[:, :-1]

y_train = tr[:, -1]

X_test = te[:, :-1]

y_test = te[:, -1]



X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model = Sequential()

model.add(LSTM(128, input_shape=(1, 1), return_sequences=True))

model.add(LSTM(64))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary() 
#from keras.utils import multi_gpu_model   #导入keras多GPU函数



#LSTM_model = multi_gpu_model(model, gpus=4) # 设置使用4个gpu

LSTM_model = model

LSTM_model.compile(loss='mean_squared_error', optimizer=Adam())



from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

# 训练

history = LSTM_model.fit(X_train, y_train, epochs=200, batch_size=200, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])
# loss曲线

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
# make predictions

trpre_LSTM = LSTM_model.predict(X_train)

tepre_LSTM = LSTM_model.predict(X_test)



# invert predictions

trpre_LSTM = scaler.inverse_transform(trpre_LSTM)

y_train_LSTM = scaler.inverse_transform([y_train])



tepre_LSTM = scaler.inverse_transform(tepre_LSTM)

y_test_LSTM = scaler.inverse_transform([y_test])



trainScore = math.sqrt(mean_squared_error(y_train_LSTM[0], trpre_LSTM[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(y_test_LSTM[0], tepre_LSTM[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
trainScore = math.sqrt(mean_squared_error(y_train_LSTM[0], trpre_LSTM[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(y_test_LSTM[0], tepre_LSTM[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
model = Sequential()

model.add(GRU(128, input_shape=(1, 1), return_sequences=True))

model.add(GRU(64))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
#from keras.utils import multi_gpu_model   #导入keras多GPU函数



#GRU_model = multi_gpu_model(model, gpus=4) # 设置使用4个gpu

GRU_model=model

GRU_model.compile(loss='mean_squared_error', optimizer=Adam())



from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

# 训练

history = GRU_model.fit(X_train, y_train, epochs=200, batch_size=200, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])
# loss曲线

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
# make predictions

trpre_GRU = GRU_model.predict(X_train)

tepre_GRU = GRU_model.predict(X_test)



# invert predictions

trpre_GRU = scaler.inverse_transform(trpre_GRU)

y_train_GRU = scaler.inverse_transform([y_train])



tepre_GRU = scaler.inverse_transform(tepre_GRU)

y_test_GRU = scaler.inverse_transform([y_test])
trainScore = math.sqrt(mean_squared_error(y_train_GRU[0], trpre_GRU[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(y_test_GRU[0], tepre_GRU[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
test.drop(axis=1, index=4319,inplace=True)
test['flow_LGBM']=test_lgb[:-1]

test['flow_LSTM']=y_test_LSTM[0]

test['flow_GRU']=y_test_GRU[0]



col = ['time', 'flow','flow_LSTM','flow_GRU','flow_LGBM','flow_SRNN']
test_0=test.head(500)
fig, ax = plt.subplots(1,1,figsize=(30, 14), dpi= 80)    

plt.plot(test_0.flow,'r-')

plt.plot(test_0.flow_LGBM,'pink')

plt.plot(test_0.flow_LSTM,'orange')

plt.plot(test_0.flow_GRU,'green')

plt.show()