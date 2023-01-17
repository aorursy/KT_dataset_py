# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')

import gc



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import matplotlib.patches as patches



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 100)



py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os,random, math, psutil, pickle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os

print(os.listdir("../input/ashrae-energy-prediction/"))
%%time

root = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## REducing memory

train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)

train_df.head()


train_df.columns.values


weather_train_df.head()


weather_train_df.columns.values
weather_test_df.head()


weather_test_df.columns.values
building_meta_df.head()
building_meta_df.columns.values


for key, d in train_df.groupby('meter_reading'):

    break

    d.head()

plt.figure(figsize = (20,5))

d['meter'].plot()
plt.figure(figsize = (15,5))

train_df['meter_reading'].plot()


train_df['meter_reading'].plot(kind='hist',

                            bins=25,

                            figsize=(15, 5),

                           title='Distribution of Target Variable (meter_reading)')

plt.show()


total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing__train_data.head(4)
# checking missing data

total = weather_train_df.isnull().sum().sort_values(ascending = False)

percent = (weather_train_df.isnull().sum()/weather_train_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_data.head(9)

weather_train_df['cloud_coverage']
weather_train_df.drop(['cloud_coverage','precip_depth_1_hr', 'sea_level_pressure'],axis=1)
total = weather_test_df.isnull().sum().sort_values(ascending = False)

percent = (weather_test_df.isnull().sum()/weather_test_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_test_data.head(9)
weather_test_df.drop(['cloud_coverage','precip_depth_1_hr', 'sea_level_pressure'],axis=1)
total = building_meta_df.isnull().sum().sort_values(ascending = False)

percent = (building_meta_df.isnull().sum()/building_meta_df.isnull().count()*100).sort_values(ascending = False)

missing_building_meta_df  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_building_meta_df.head(6)
train_df.dtypes.value_counts()
train_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
correlations = train_df.corr()['meter_reading'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
# weather correlation

w_correlations = weather_train_df.corr()['wind_speed'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', w_correlations.tail(15))

print('\nMost Negative Correlations:\n',w_correlations.head(15))
corrs = train_df.corr()

corrs
w_corrs = weather_train_df.corr()

w_corrs
train_df.building_id.nunique()
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm   # 进度条工具



import os



for dirname, _, filenames in os.walk('data_set'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

root = '../input/ashrae-energy-prediction/'

building_df = pd.read_csv(root+"building_metadata.csv")

weather_train = pd.read_csv(root+"weather_train.csv")

train = pd.read_csv(root+"train.csv")



# 数据关联

train = train.merge(building_df, left_on="building_id", right_on="building_id", how="left")

train = train.merge(weather_train, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])



print(train.head(5))



# 特征处理

# 天气时间字段处理

train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"] = train["timestamp"].dt.day

train["year"] = train["timestamp"].dt.year

train["weekend"] = train["timestamp"].dt.weekday

train["month"] = train["timestamp"].dt.month

del train["timestamp"]



# 建筑数据统一范围

train['year_built'] = train['year_built'] - 1900

train['square_feet'] = np.log(train['square_feet'])



# 特征转换

le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])



# 特征选取与整理

categoricals = ["site_id", "building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]

drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage", "dew_temperature"]

feat_cols = categoricals + numericals

target = np.log1p(train["meter_reading"])  # 数据平滑处理

del train["meter_reading"]

train = train.drop(drop_cols, axis=1)



# lightbgm参数设置

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'subsample': 0.2,

    'learning_rate': 0.1,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9,

    'alpha': 0.1,

    'lambda': 0.1

}



folds = 3

seed = 666

kf = KFold(n_splits=folds, shuffle=True, random_state=seed)



models = []

for train_index, val_index in kf.split(train):

    train_X = train[feat_cols].iloc[train_index]

    val_X = train[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                    lgb_train,

                    num_boost_round=300,

                    valid_sets=(lgb_train, lgb_eval),

                    early_stopping_rounds=100,

                    verbose_eval=100)

    models.append(gbm)



# 垃圾处理机制，collect(),返回释放掉的资源个数

import gc



del train, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target

gc.collect()



# 测试集数据处理

test = pd.read_csv(root+"test.csv")

test = test.merge(building_df, left_on="building_id", right_on="building_id", how="left")

del building_df

gc.collect()

test["primary_use"] = le.transform(test["primary_use"])

weather_test = pd.read_csv(root+"weather_test.csv")

weather_test = weather_test.drop(drop_cols, axis=1)

test = test.merge(weather_test, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"], how="left")

del weather_test

test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)

test["year"] = test["timestamp"].dt.year.astype(np.uint16)

test["day"] = test["timestamp"].dt.day.astype(np.uint8)

test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)

test["month"] = test["timestamp"].dt.month.astype(np.uint8)

test['year_built'] = test['year_built'] - 1900

test['square_feet'] = np.log(test['square_feet'])



test = test[feat_cols]



i = 0

res = []

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0] / 50000)))):

    res.append(np.expm1(sum([model.predict(test.iloc[i:i + step_size]) for model in models]) / folds))

    i += step_size



res = np.concatenate(res)

submission = pd.read_csv(root+'sample_submission.csv')

submission['meter_reading'] = res

submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)