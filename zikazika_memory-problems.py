import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge, RidgeCV

import gc

from catboost import CatBoostRegressor

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import gc
%%time

train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
import dask

import dask.dataframe as dd
%%time

train1 = dd.read_csv("../input/LANL-Earthquake-Prediction/train.csv",  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train.head()
plot1=train1[0:150000].time_to_failure

plot2=train1[0:150000].acoustic_data

%%time

plt.figure(figsize=(20,5))

plt.plot(-plot1.compute(), plot2.compute());

plt.xlabel("- Quaketime")

plt.ylabel("Signal")

plt.title("PLOT 0");
%%time

plt.figure(figsize=(20,5))

plt.plot(-train.time_to_failure.values[0:150000], train.acoustic_data.values[0:150000]);

plt.xlabel("- Quaketime")

plt.ylabel("Signal")

plt.title("PLOT 1");
from IPython.display import Image

Image("../input/example/example.png")
del train1

gc.collect()
def reduce_mem_usage(df):



    for col in df.columns:

        col_type = df[col].dtype.name



        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:

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



    return df
train=reduce_mem_usage(train)

gc.collect()
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time_to_failure. 150 000 rows")

plt.plot(train['acoustic_data'].values[:15000000], color='y')

ax1.set_ylabel('acoustic_data', color='y')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train['time_to_failure'].values[:15000000], color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)
fig, ax = plt.subplots(1,2, figsize=(20,5))

sns.distplot(train['acoustic_data'].values[:15000000], ax=ax[0], color="Yellow", bins=100)

ax[0].set_xlabel("acoustic_data")

ax[0].set_ylabel("Density")

ax[0].set_title("acoustic_data distribution")





sns.distplot(train['time_to_failure'].values[:15000000], ax=ax[1], color="Red", bins=100)

ax[1].set_xlabel("time_to_failure")

ax[1].set_ylabel("Density")

ax[1].set_title("time_to_failure distribution");
min_ = train.acoustic_data.mean() - 3 * train.acoustic_data.std()

max_ = train.acoustic_data.mean() + 3 * train.acoustic_data.std() 
#train["sharp_rise"] = np.where((train.acoustic_data >= min_) & (train.acoustic_data <= max_), 100, 0)