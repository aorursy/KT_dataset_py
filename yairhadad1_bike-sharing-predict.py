# base lib

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



# set seaborn

sns.set()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
print("data nan Count:")

print('Nan in data:\n',train.isnull().sum())
train['datetime_hour'] = pd.DatetimeIndex(train['datetime']).hour

train['datetime_day'] = pd.DatetimeIndex(train['datetime']).day

train['datetime_dayofweek'] = pd.DatetimeIndex(train['datetime']).dayofweek

train['datetime_month'] = pd.DatetimeIndex(train['datetime']).month

train['datetime_weak'] = pd.DatetimeIndex(train['datetime']).week

train['datetime_year'] = pd.DatetimeIndex(train['datetime']).year
train.groupby('datetime_weak').mean()['count'].plot.bar(figsize=(14,5), title = "Weak - Count mean")

plt.show()



train.groupby('datetime_hour').mean()['count'].plot.bar(figsize=(14,5), title = "Hour - Count mean")

plt.axhline(train['count'].mean(),color = 'b',linestyle='--')

plt.show()
train.groupby('datetime_year').mean()['count'].plot.bar(title = "year - Count mean")

plt.show()
print(train.groupby('season').size())

train.groupby('season').mean()['count'].plot.bar(title = "Season - Count mean")

plt.show()





# in sprint and summer more use
train.groupby('atemp').mean()['count'].plot(c = 'r', title = "temp - G , atemp - R")

train.groupby('temp').mean()['count'].plot(c = 'g')

plt.show()



plt.title('ATemp - Temp')

sns.kdeplot(train['atemp'], bw=.2)

sns.kdeplot(train['temp'], bw=2)

plt.legend();

plt.show()



temp_pca = PCA(n_components = 1)

train["temp_pca"] = temp_pca.fit_transform(train[['atemp','temp']])



sns.kdeplot(train['temp_pca'], bw=2)

plt.show()
print(train.groupby('weather').count()['count'])

# maybe we what to remove 4 ? or get more data from 4.



# For exmple:

# train.groupby('weather').mean()['count'].plot.bar()

# plt.show()



sns.distplot(train['weather']);
train.groupby('humidity').mean()['count'].plot()

plt.axvline(17,color = 'r',linestyle='--')

plt.show()



# we can see that the count start in 17 and start to go down from there



sns.distplot(train['humidity']);
print(train.groupby('windspeed').count()['count'])

train.groupby('windspeed').count()['count'].plot(title = "windspeed count")

plt.show()



train.groupby('windspeed').mean()['count'].plot(title = "windspeed count mean")

plt.show()



# maybe it better to remove 31.0009 and above (very small data)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,15))

row, colums = 2, 4



plt.subplot(colums, row, 1)

sns.lineplot(x="casual", y="count", data=train)



plt.subplot(colums, row, 2)

sns.lineplot(x="registered", y="count", data=train)



plt.subplot(colums, row, 3)

sns.distplot(train['casual'])



plt.subplot(colums, row, 4)

sns.distplot(train['registered'])



plt.subplot(colums, row, 5)

sns.boxplot(train['casual'])



plt.subplot(colums, row, 6)

sns.boxplot(train['registered'])



plt.subplot(colums, row, 7)

train.groupby('datetime_hour').mean()['casual'].plot.bar()



plt.subplot(colums, row, 8)

train.groupby('datetime_hour').mean()['registered'].plot.bar()



plt.show()

# we can't use this data for module - this is a memory leak (we dont have this data in the field.)

# but maybe make 2 models , one for casual and one for registered
train.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
plt.figure(figsize=(10,5))

sns.pointplot(x=train["datetime_hour"],

              y=train["count"],

              hue=train["season"], 

              data=train)

plt.show()
# Perper data

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin , RegressorMixin

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error # for RMSLE

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM , Embedding

from keras.optimizers import RMSprop

from keras import backend as K

import keras



import tensorflow as tf

import lime

import lime.lime_tabular

from sklearn.metrics import make_scorer
# transformers:



# get selected colums

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]



    

# Change datetime to day, week, month and year

class DatetimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        timedata = data_dict['datetime']

        

        day = pd.DatetimeIndex(timedata).day

        hour = pd.DatetimeIndex(timedata).hour

        week = pd.DatetimeIndex(timedata).week

        month = pd.DatetimeIndex(timedata).month

        year = pd.DatetimeIndex(timedata).year

        timestamp = pd.DatetimeIndex(timedata).astype(np.int64)



        data_dict['datetime_hour'] = hour

        data_dict['datetime_day'] = day

        data_dict['datetime_weak'] = week

        data_dict['datetime_month'] = month

        data_dict['datetime_year'] = year

        data_dict['datetime_timestape'] = timestamp

        

        return data_dict

    

class DummyEncoder(BaseEstimator, TransformerMixin):



    def __init__(self, n_values='auto', categories = 'auto'):

        self.n_values = n_values

        self.categories = categories



    def transform(self, X):

        ohe = OneHotEncoder(categories = self.categories,sparse=False, n_values=self.n_values)

        return ohe.fit_transform(X)[:,:-1]



    def fit(self, X, y=None, **fit_params):

        return self

X = train.drop(['count','casual','registered'],axis = 1)

y = train['count']



prdict_feature = ['datetime','season','holiday', 'windspeed' ,

                  'workingday','weather','temp' , 'atemp', 'humidity']

X = X[prdict_feature]

 

X_train_base, X_test_base, y_train, y_test = train_test_split(X, y, shuffle= False)

_, _, y_train_registered, y_test_registered = train_test_split(X, train['registered'], shuffle= False)

_, _, y_train_casual, y_test_casual = train_test_split(X, train['casual'], shuffle= False)
# preper data

preper_data_pipeline = Pipeline([('time_transformer',DatetimeTransformer())])



# one hot

label_pipeline = Pipeline([('hot_columns', ItemSelector(key = ['datetime_year'])),

                             ('label_encoder', OneHotEncoder())])



label_pipeline_weather = Pipeline([('weather_columns', ItemSelector(key = ['weather'])),

                             ('label_encoder', DummyEncoder(categories = [[1,2,3,4]]))])



label_pipeline_season = Pipeline([('season_columns', ItemSelector(key = ['season'])),

                             ('label_encoder', DummyEncoder(categories = [[1,2,3,4]]))])



# min max

min_max_pipeline = Pipeline([('min_max_columns', ItemSelector(key = ['datetime_hour','datetime_day','datetime_weak',

                                                                     'datetime_month','datetime_timestape','humidity',

                                                                    'workingday','holiday'])),

                             ('minMaxScaler', MinMaxScaler())])



# temp

temp_pipeline = Pipeline([("temp_columns", ItemSelector(key = ['temp','atemp'])),

                         ("pca", PCA(n_components = 1)),

                         ("min_max", MinMaxScaler())])



# windspeed

windspeed_pipline = Pipeline([('windspeed_columns',ItemSelector(key = ['windspeed'])),

                             ('kbins',KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')),

                             ("min_max", MinMaxScaler())])





feature_union = FeatureUnion([('label', label_pipeline),

                              ('label_pipeline_weather',label_pipeline_weather),

                              ('label_pipeline_season',label_pipeline_season),

                              ('windspeed',windspeed_pipline),

                              ('min_max', min_max_pipeline),

                              ('temp', temp_pipeline)])



# full feature pipline

feature_pipeline = Pipeline([('preper_data',preper_data_pipeline),

                            ('feature_union',feature_union)])





feature_pipeline.fit(X_train_base)



X_train = feature_pipeline.transform(X_train_base)

X_test = feature_pipeline.transform(X_test_base)
def rmsle(y_true,y_pred):

    diff=np.log(y_pred+1)-np.log(y_true+1)

    mean_error = np.square(diff).mean()

    return np.sqrt(mean_error)



def rmsle_xgb(predictions, dmat):

    labels = dmat.get_label()

    diffs = np.log(predictions + 1) - np.log(labels + 1)

    squared_diffs = np.square(diffs)

    avg = np.mean(squared_diffs)

    return ('RMSLE', np.sqrt(avg))



def rmsle_K(y, y0):

    return K.sqrt(K.square(tf.log1p(y) - tf.log1p(y0)))

%%time

params = {

    'max_depth': 5, 

    'eta': .01,

    'booster' : 'dart',

    'subsample': 0.8, 

    'objective':'count:poisson',

    'eval_metric':'rmse',

    'silent': 1

}



matrix_train = xgb.DMatrix(X_train,label=y_train_registered)

matrix_test = xgb.DMatrix(X_test,label=y_test_registered)

reg_xgb_registered = xgb.train(params=params,

                    dtrain=matrix_train,num_boost_round=5000, 

                    early_stopping_rounds=100,verbose_eval=100,

                    feval = rmsle_xgb,

                    evals=[(matrix_test,'test')])
%%time

# result = xgb.cv(params=params, dtrain=matrix_train, num_boost_round=5000, early_stopping_rounds=50, feval = rmsle_xgb)

# XGB_Cross_train_registered = result['train-RMSLE-mean'].mean()



prdict_result = reg_xgb_registered.predict(xgb.DMatrix(X_train), ntree_limit = reg_xgb_registered.best_ntree_limit)

XGB_Score_train_registered = rmsle(y_train_registered, prdict_result)



prdict_result = reg_xgb_registered.predict(xgb.DMatrix(X_test), ntree_limit = reg_xgb_registered.best_ntree_limit)

XGB_Score_test_registered = rmsle(y_test_registered, prdict_result)



# print("xgb train cross:",XGB_Cross_train_registered)

print("xgb train:",XGB_Score_train_registered)

print("xgb test:",XGB_Score_test_registered)
%%time

params = {

    'max_depth': 5, 

    'eta': .01,

    'booster' : 'dart',

    'subsample': 0.8, 

    'objective':'count:poisson',

    'eval_metric':'rmse',

    'silent': 1

}



matrix_train = xgb.DMatrix(X_train,label=y_train_casual)

matrix_test = xgb.DMatrix(X_test,label=y_test_casual)

reg_xgb_casual = xgb.train(params=params,

                    dtrain=matrix_train,num_boost_round=5000, 

                    early_stopping_rounds=100,verbose_eval=100,

                    feval = rmsle_xgb,

                    evals=[(matrix_test,'test')])
%%time

# result = xgb.cv(params=params, dtrain=matrix_train, num_boost_round=5000, early_stopping_rounds=50, feval = rmsle_xgb)

# XGB_Cross_train_casual = result['train-RMSLE-mean'].mean()



prdict_result = reg_xgb_casual.predict(xgb.DMatrix(X_train), ntree_limit = reg_xgb_casual.best_ntree_limit)

XGB_Score_train_casual = rmsle(y_train_casual, prdict_result)



prdict_result = reg_xgb_casual.predict(xgb.DMatrix(X_test), ntree_limit = reg_xgb_casual.best_ntree_limit)

XGB_Score_test_casual = rmsle(y_test_casual, prdict_result)



# print("xgb train cross:",XGB_Cross_train_casual)

print("xgb train:",XGB_Score_train_casual)

print("xgb test:",XGB_Score_test_casual)
class XGB_Combin_regressor(BaseEstimator, RegressorMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y):

        return self



    def predict(self, X):

        registered_predict = reg_xgb_registered.predict(xgb.DMatrix(X), ntree_limit = reg_xgb_registered.best_ntree_limit)

        registered_casual = reg_xgb_casual.predict(xgb.DMatrix(X), ntree_limit = reg_xgb_casual.best_ntree_limit)

        return (registered_predict + registered_casual).round()
count_reg = XGB_Combin_regressor()

count_reg.fit(X_train,y_train) 



train_score = rmsle(y_train, count_reg.predict(X_train).round())

test_score = rmsle(y_test, count_reg.predict(X_test).round())



print("train_score:",train_score)

print("test_score:",test_score)
# full estimator



test_transform = feature_pipeline.transform(test)

predict = count_reg.predict(test_transform).round()

datetimecol = test["datetime"]