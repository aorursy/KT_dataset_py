import gc

import os

import random

import pickle

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb



from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder



path_data = "/kaggle/input/ashrae-energy-prediction/"

path_train = path_data + "train.csv"

path_test = path_data + "test.csv"

path_building = path_data + "building_metadata.csv"

path_weather_train = path_data + "weather_train.csv"

path_weather_test = path_data + "weather_test.csv"



plt.style.use("seaborn")

sns.set(font_scale=1)



myfavouritenumber = 0

seed = myfavouritenumber

random.seed(seed)
%time df_train = pd.read_csv(path_train)

%time df_test = pd.read_csv(path_test)

%time building = pd.read_csv(path_building)

%time weather_train = pd.read_csv(path_weather_train)

%time weather_test = pd.read_csv(path_weather_test)
## Memory optimization



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
df_train = reduce_mem_usage(df_train, use_float16=True)

df_test = reduce_mem_usage(df_test,use_float16=True)

building = reduce_mem_usage(building, use_float16=True)

weather_train = reduce_mem_usage(weather_train, use_float16=True)

weather_test = reduce_mem_usage(weather_test, use_float16=True)
df_train = df_train.merge(building, on="building_id", how="left")

df_train = df_train.merge(weather_train, on=["site_id", "timestamp"], how="left")

df_test = df_test.merge(building, on="building_id", how="left")

df_test = df_test.merge(weather_test, on=["site_id", "timestamp"], how="left")

del weather_train,weather_test;gc.collect()
df_train.shape,df_test.shape
df_train.dtypes
df_train.describe()
# df_train = df_train.drop("building_id",axis=1)
plt.figure(figsize = (15,5))

df_train['meter_reading'].plot()
df_train['meter_reading'].plot(kind='hist',

                              bins=15,

                              figsize=(15,5),

                              title='distribution of "meter_reading"')
# sort data by "timestamp"

df_train = df_train.sort_values(by="timestamp" , ascending=True) 
correlation = df_train.corr()

correlation['meter_reading'].sort_values()
correlation
plt.figure(figsize = (15,10))

sns.heatmap(correlation,cmap=plt.cm.RdYlBu_r,vmin=-0.25,

            annot=True,vmax=0.6)
# building_id

print('the number of building_id:{}'.format(df_train.building_id.nunique()))
meter1 = df_train['meter_reading'].loc[df_train.meter==0].groupby(df_train.building_id).mean()

meter2 = df_train['meter_reading'].loc[df_train.meter==1].groupby(df_train.building_id).mean()

meter3 = df_train['meter_reading'].loc[df_train.meter==2].groupby(df_train.building_id).mean()

meter4 = df_train['meter_reading'].loc[df_train.meter==3].groupby(df_train.building_id).mean()
plt.figure(figsize=(20,5))

meter1.plot(kind='line',logy=True,color='green')

meter2.plot(kind='line',color='blue')

meter3.plot(kind='line',color='yellow')

meter4.plot(kind='line',color='red')

plt.legend(['electricity','chilledwater','steam','hotwater'])
df_train['meter_reading'].groupby(df_train.meter).mean()
len(df_train.building_id)
build_meter = df_train.meter.groupby(df_train.building_id).nunique()
df_train.meter_reading.loc[df_train.building_id==5].sum()
x = [0,0,0,0]

num_x = [0,0,0,0]

for i in range(len(build_meter)):

    heat = df_train.meter_reading.loc[df_train.building_id==i].sum()

    x[build_meter[i]-1] = x[build_meter[i]-1] + heat

    num_x[build_meter[i]-1] = num_x[build_meter[i]-1]+1
pd.DataFrame(x).plot(kind='bar',title='influence(max) of the numeber of meter categories',xticks=[1,2,3,4])

for i in range(4):

    x[i] = x[i]/num_x[i]

pd.DataFrame(x).plot(kind='bar',title='influence(mean by building num) of the numeber of meter categories',xticks=[1,2,3,4])
def plot_col(column):

    plt.subplots(figsize=(6,6))

    sns.distplot(df_train[column],color='green').set_title(column)

    sns.distplot(df_test[column],color='yellow').set_title(column)

    plt.legend(['train','test'])
# meter

plt.subplot(2,2,1)

df_train['meter'].plot(kind='hist',bins=4,figsize=(16,5),xticks=[0,1,2,3])

plt.title("the number of meter(0: electricity, 1: chilledwater, 2: steam, 3: hotwater)")

ave = []

for i in range(4):

    ave.append(df_train.loc[(df_train.meter==i)].meter_reading.mean())

plt.subplot(2,2,2,title='the mean of meter_reading')

plt.plot(ave)

plt.subplot(2,2,3,title='the number of meter_reading=0')

ave = []

for i in range(4):

    ave.append(df_train.meter_reading.loc[(df_train.meter==i) & (df_train.meter_reading==0)].count())

plt.plot(ave)
# site_id

plt.subplot(1,2,1)

df_train['site_id'].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(14):

    ave.append(df_train.loc[(df_train.site_id==i)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(ave)
# square_feet - Gross floor area of the building

plt.subplot(1,2,1)

df_train['square_feet'].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(8):

    ave.append(df_train.loc[(df_train.square_feet/100000<i+1) & (df_train.square_feet/100000>=i)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(ave)

plot_col('square_feet')
plt.subplot(1,2,1)

df_train['year_built'].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(10):

    ave.append(df_train.loc[(df_train.year_built>1900+i*10) & (df_train.year_built<=1910+i*10)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(ave)

print("number of nan in 'year_build':{},pencent:{}%".format(df_train.meter_reading.loc[df_train.year_built.isnull()].count(),100*df_train.meter_reading.loc[df_train.year_built.isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when 'year_build' is nan:{}".format(df_train.meter_reading.loc[df_train.year_built.isnull()].mean()))
df_train['year_built_ifnan'] = df_train.year_built.isnull().astype('int')

df_train['year_built'] = df_train['year_built'].fillna(2018)
# floor_count

plt.subplot(1,2,1)

df_train['floor_count'].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(10):

    ave.append(df_train.loc[(df_train.floor_count>i*2.5) & (df_train.floor_count<2.5+i*2.5)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(ave)

print("number of nan in 'floor_count':{},pencent:{}%".format(df_train.meter_reading.loc[df_train.floor_count.isnull()].count(),100*df_train.meter_reading.loc[df_train.floor_count.isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when 'floor_count' is nan:{}".format(df_train.meter_reading.loc[df_train.floor_count.isnull()].mean()))
df_train['floor_count_ifnan'] = df_train.floor_count.isnull().astype('int')

df_train = df_train.drop('floor_count',axis=1)
# air_temperature

feature = "air_temperature"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(10):

    ave.append(df_train.loc[(df_train[feature]>-30+i*10) & (df_train[feature]<-20+i*10)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(-25,75,10),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train["air_temperature"] = df_train["air_temperature"].fillna(35)
# cloud_coverage

feature = "cloud_coverage"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

plt.title('Portion of the sky covered in clouds, in oktas')

ave = []

for i in [0,2,4,6,8]:

    ave.append(df_train.loc[(df_train[feature]==i)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(0,10,2),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train['cloud_coverage_ifnan'] = df_train.cloud_coverage.isnull().astype("int")

df_train['cloud_coverage'] = df_train.cloud_coverage.fillna(5)
df_train['primary_use_encoded'] = LabelEncoder().fit_transform(df_train.primary_use).astype("int")
# primary_use

feature = "primary_use_encoded"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

plt.title('Indicator of the primary category of activities for the building')

ave = []

for i in range(16):

    ave.append(df_train.loc[(df_train[feature]==i)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(0,16,1),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
primary_use_num = {}

for i in range(16):

    primary_use_num[i] = df_train.meter_reading.loc[df_train.primary_use_encoded==i].count()

df_train['primary_use_encoded'] = df_train.primary_use_encoded.map(primary_use_num)

df_train = df_train.drop('primary_use',axis=1)
primary_use_num
# dew_temperature

feature = "dew_temperature"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(10):

    ave.append(df_train.loc[(df_train[feature]>-29+i*6) & (df_train[feature]<-23+i*6)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(-26,30,6),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train['dew_temperature'] = df_train['dew_temperature'].fillna(23)
# precip_depth_1_hr

feature = "precip_depth_1_hr"

plt.figure(figsize=(16,5))

ave = []

num = []

for i in range(9):

    ave.append(df_train.loc[(df_train[feature]>-50+i*50) & (df_train[feature]<0+i*50)].meter_reading.mean())

    num.append(df_train.loc[(df_train[feature]>-50+i*50) & (df_train[feature]<0+i*50)].meter_reading.count())

plt.subplot(1,2,1)

plt.title('mean of meter_reading')

plt.bar(np.arange(-50,400,50),ave,width=40)

plt.subplot(1,2,2)

plt.title('number of precip_depth_1_hr')

plt.bar(np.arange(-50,400,50),num,width=40)

plt.show()

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))

print('number of 0:{}'.format(df_train.loc[(df_train[feature]==0)].meter_reading.count()))

print("mean of 'meter_reading when 'number is 0:{}".format(df_train.loc[(df_train[feature]==0)].meter_reading.mean()))
df_train['precip_depth_1_hr_ifnan'] = df_train.precip_depth_1_hr.isnull().astype("int")

df_train['precip_depth_1_hr'] = df_train['precip_depth_1_hr'].fillna(300)

# I do not know why how to create a new column "precip_depth_1_hr_ifzero"

# df_train['precip_depth_1_hr_ifzero'] = df_train.apply(lambda x:0 if x.precip_depth_1_hr==0 else 1,axis=1)
# sea_level_pressure

feature = "sea_level_pressure"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(8):

    ave.append(df_train.loc[(df_train[feature]>970+i*10) & (df_train[feature]<980+i*10)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(970,1050,10),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train['sea_level_pressure'] = df_train['sea_level_pressure'].fillna(980)
# wind_direction

feature = "wind_direction"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(12):

    ave.append(df_train.loc[(df_train[feature]>=i*30) & (df_train[feature]<30+i*30)].meter_reading.mean())

plt.subplot(1,2,2)

plt.plot(np.arange(0,360,30),ave)

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train['wind_direction'] = df_train['wind_direction'].fillna(0)
# wind_spped

feature = "wind_speed"

plt.subplot(1,2,1)

df_train[feature].plot(kind='hist',figsize=(16,5))

ave = []

for i in range(7):

    ave.append(df_train.loc[(df_train[feature]>=i*2.5) & (df_train[feature]<2.5+i*2.5)].meter_reading.mean())

plt.subplot(1,2,2)

plt.bar(np.arange(0,17.5,2.5),ave,width=1.5)

plt.title('mean of meter_reading')

print("number of nan in '{}':{},pencent:{}%".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].count(),100*df_train.meter_reading.loc[df_train[feature].isnull()].count()/len(df_train)))

print("mean of 'meter_reading' when '{}' is nan:{}".format(feature,df_train.meter_reading.loc[df_train[feature].isnull()].mean()))
df_train['wind_speed'] = df_train['wind_speed'].fillna(15)
# timestamp

df_train.timestamp = pd.to_datetime(df_train.timestamp, format="%Y-%m-%d %H:%M:%S")

df_train["hour"] = df_train.timestamp.dt.hour

df_train['year'] = df_train['timestamp'].dt.year

df_train['month'] = df_train['timestamp'].dt.month

df_train['day'] = df_train['timestamp'].dt.day

df_train["weekday"] = df_train.timestamp.dt.weekday

df_train['age'] = (df_train['year'] - df_train['year_built'])

df_train = df_train.drop('timestamp',axis=1)
df_train = reduce_mem_usage(df_train)
# import lightgbm as lbt



# fraction = 0.8

# y_train = np.log1p(df_train['meter_reading'][0:int(fraction*len(df_train))])

# X_train = df_train.drop('meter_reading',axis=1)[0:int(fraction*len(df_train))]

# y_valid = np.log1p(df_train['meter_reading'][int(fraction*len(df_train)):])

# X_valid = df_train.drop('meter_reading',axis=1)[int(fraction*len(df_train)):]

# len(X_train),len(X_valid)
# df_train.columns
# import lightgbm as lgb



# categorical_features = ['year_built_ifnan', 'floor_count_ifnan',

#        'cloud_coverage_ifnan', 'primary_use_encoded',

#        'precip_depth_1_hr_ifnan', 'hour', 'year', 'month', 'day', 'weekday']

# train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)

# valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features, free_raw_data=False)





# params = {

#     "objective": "regression",

#     "boosting": "gbdt",#dart,gbdt

#     "num_leaves": 45,

#     "learning_rate": 0.05,

#     "feature_fraction": 0.9,

#     "reg_lambda": 2,

#     "metric": "rmse"

# }



# model = lgb.train(params, train_set=train_set, num_boost_round=2000, valid_sets=valid_set, verbose_eval=200, early_stopping_rounds=200)

# model.save_model('/kaggle/working/lightgbm_cat.txt')

# # del X_train,y_train,X_valid,y_train,train_set,valid_set

# # gc.collect()
del _i1,_i2,_i3,_i4,_i5,_i6,_i7,_i8,_i9,_i10;

del _i11,_i12,_i13,_i14,_i15,_i16,_i17,_i18,_i19,_i20;

del _i21,_i22,_i23,_i24,_i25,_i26,_i27,_i28,_i29,_i30;

del _i31,_i32,_i33,_i34,_i35,_i36,_i37,_i38,_i39,_i40;

# del X_train,y_train,X_valid,y_valid;

gc.collect()
# X_train = df_train.drop('meter_reading',axis=1)

# y_train = np.log1p(df_train.meter_reading)
# X_half_1 = X_train[:int(X_train.shape[0] / 2)]

# X_half_2 = X_train[int(X_train.shape[0] / 2):]

# y_half_1 = y_train[:int(y_train.shape[0] / 2)]

# y_half_2 = y_train[int(y_train.shape[0] / 2):]

# categorical_features = ['year_built_ifnan', 'floor_count_ifnan',

#        'cloud_coverage_ifnan', 'primary_use_encoded',

#        'precip_depth_1_hr_ifnan', 'hour', 'year', 'month', 'day', 'weekday']



# d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categorical_features, free_raw_data=False)

# d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categorical_features, free_raw_data=False)

# watchlist_1 = [d_half_2, d_half_1]

# watchlist_2 = [d_half_1, d_half_2]

# params = {

#     "objective": "regression",

#     "boosting": "gbdt",#dart,gbdt

#     "num_leaves": 45,

#     "learning_rate": 0.02,

#     "feature_fraction": 0.9,

#     "reg_lambda": 2,

#     "metric": "rmse"

# }
# print("Building model with first half and validating on second half:")

# model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)



# print("Building model with second half and validating on first half:")

# model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)
# model_half_1.save_model('/kaggle/working/model_half_1.txt')

# model_half_2.save_model('/kaggle/working/model_half_2.txt')
# df_train
# fraction = 0.8

# y_train = np.log1p(df_train['meter_reading'][0:int(fraction*len(df_train))])

# X_train = df_train.drop('meter_reading',axis=1)[0:int(fraction*len(df_train))]

# y_valid = np.log1p(df_train['meter_reading'][int(fraction*len(df_train)):])

# X_valid = df_train.drop('meter_reading',axis=1)[int(fraction*len(df_train)):]

# len(X_train),len(X_valid)
# X_train = np.reshape(X_train.values,[-1,1,22])

# X_valid = np.reshape(X_valid.values,[-1,1,22])

# y_train = np.reshape(y_train.values,[-1,1,1])

# y_valid = np.reshape(y_valid.values,[-1,1,1])
# from keras import Sequential

# from keras.preprocessing.sequence import pad_sequences

# from sklearn.model_selection import train_test_split

# from keras.models import Sequential, Model

# from keras.layers import LSTM, Dense, Bidirectional, Input, Dropout, BatchNormalization

# from keras import backend as K

# from keras.engine.topology import Layer

# from keras import initializers, regularizers, constraints
# from keras import backend as K

# def estimate(y_valid,y_pred):

#     l = K.int_shape(y_pred)

#     return K.pow(K.sum(K.pow(K.log(y_valid+1)-K.log(y_pred+1),2)),0.5)

# def loss(y_valid,y_pred):

#     l = K.int_shape(y_pred)

#     return K.pow(K.sum(K.pow(K.log(y_valid+1)-K.log(y_pred+1),2)),0.5)

# model = Sequential()

# model.add(BatchNormalization(input_shape=(1,22)))

# model.add(Bidirectional(LSTM(32,dropout=0.4,recurrent_dropout=0.4,activation='tanh',return_sequences=True)))

# model.add(Bidirectional(LSTM(64,return_sequences=True)))

# model.add(BatchNormalization(input_shape=(1,32)))

# model.add(Bidirectional(LSTM(128, activation='tanh',return_sequences=True)))

# model.add(BatchNormalization(input_shape=(1,64)))

# model.add(Dense(512,activation='relu'))

# model.add(Dropout(0.3))

# model.add(Dense(512,activation="relu"))

# model.add(Dropout(0.3))

# model.add(Dense(1,activation="softmax"))

# model.compile(loss='mse',optimizer='adam',metrics=['mse'])

# print(model.summary())
# history = model.fit(X_train,y_train,batch_size=300,epochs=15,

#                    validation_data=(X_valid,y_valid))
# model.save('/kaggle/working/LSTM.h5')
# # Plot training & validation accuracy values

# plt.plot(history.history['mse'])

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_mse'])

# plt.plot(history.history['val_loss'])

# plt.title('Model accuracy')

# plt.ylabel('Accuracy')

# plt.xlabel('Epoch')

# plt.legend(['estimate', 'loss','val_estimate','val_loss'], loc='upper left')

# plt.show()