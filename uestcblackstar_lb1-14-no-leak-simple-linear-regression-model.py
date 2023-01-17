import numpy as np

import pandas as pd 

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import holidays

import time

from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)

import gc

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

%matplotlib inline
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

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
bd_df = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae/building_metadata.csv"))

wt_trn = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae/weather_train.csv"))

trn = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae/train.csv"))

ts = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae/test.csv"))

wt_ts = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae/weather_test.csv"))
def add_time(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)

    df["month"] = df["timestamp"].dt.month.astype(np.int8)

    df["weekday"] = df["timestamp"].dt.weekday.astype(np.int8)

    return df

trn = add_time(trn)

ts = add_time(ts)
def add_diff(df):

    df['tem_diff'] = df['air_temperature'] - df['dew_temperature']

    return df

wt_trn = add_diff(wt_trn)

wt_ts = add_diff(wt_ts)
def tran_pca(df):

    pca = PCA(n_components = 2)

    tmp = pca.fit_transform(df[['air_temperature', 'cloud_coverage', 'precip_depth_1_hr', 'wind_speed', 'wind_direction', 'tem_diff']])

    tmp = pd.DataFrame(tmp, columns = ['component1', 'component2'])

    df = pd.concat([df, tmp], axis=1, sort=False)

    return df

wt_trn = tran_pca(wt_trn)

wt_ts = tran_pca(wt_ts)
wt_trn["timestamp"] = wt_trn["timestamp"].astype(str)

wt_ts["timestamp"] = wt_ts["timestamp"].astype(str)

trn["timestamp"] = trn["timestamp"].astype(str)

ts["timestamp"] = ts["timestamp"].astype(str)



trn = pd.merge(trn, bd_df, how='left', on=['building_id'], suffixes=(False,False))

trn = pd.merge(trn, wt_trn, on = ['site_id', 'timestamp'], suffixes=(False, False))



ts = pd.merge(ts, bd_df, how='left', on=['building_id'], suffixes=(False,False))

ts = pd.merge(ts, wt_ts, on = ['site_id', 'timestamp'], suffixes=(False, False))
def season_encode(df):

    df['summer'] = 0

    df.loc[df.query('month in [6,7,8,9]').index,'summer'] = 1

    df['winter'] = 0

    df.loc[df.query('month in [11,12,1,2]').index,'winter'] = 1

    df['summer'] = df['summer'].astype(np.int8)

    df['winter'] = df['winter'].astype(np.int8)

    return df

trn = season_encode(trn)

ts = season_encode(ts)
def if_weekend(df):

    df['M_F'] = 0

    df.loc[df.query('weekday in [5,6]').index, 'M_F'] = 1

    df['M_F'] = df['M_F'].astype(np.int8)

    return df

trn = if_weekend(trn)

ts = if_weekend(ts)
trn["meter_reading"] = trn["meter_reading"].clip(upper = trn["meter_reading"].quantile(.975))

trn['meter_reading'] = np.log1p(trn['meter_reading'])
trn = trn.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
trn = trn[['building_id', 'meter', 'hour', 'M_F',  'winter', 'component1', 'component2', 'summer', 'meter_reading']]

ts = ts[['building_id', 'meter', 'hour', 'M_F', 'winter', 'component1', 'component2', 'summer']]
sample_time = 1

y_pred = np.zeros(len(ts))

score = 0

for i in range(sample_time):

    models = []

    trn_group = trn.groupby(['building_id', 'meter', 'hour','M_F'])

    ts_group = ts.groupby(['building_id', 'meter', 'hour', 'M_F'])

    for group in tqdm(trn_group.groups.keys()):

        df = trn_group.get_group(group)

#         if len(df)>20:

#             df = df.sample(frac=0.5, random_state=i)

        X = pd.concat([

                        df['component1'], 

                        df['component2'],

                       df['summer'],

                       df['winter'],

                       df['component1']*df['summer'], 

                       df['component1']*df['winter']],axis=1)

        y = df['meter_reading']

        scaler = MinMaxScaler()

        scaler.fit(X)

        X = scaler.transform(X)

        model = LinearRegression(n_jobs=-1)

        model.fit(X, y)

        

        df = ts_group.get_group(group)

        X = pd.concat([

                        df['component1'], 

                        df['component2'],

                       df['summer'],

                       df['winter'],

                       df['component1']*df['summer'], 

                       df['component1']*df['winter']],axis=1)

        idx = X.index

        X = scaler.transform(X)

        tmp = model.predict(X)/sample_time

        tmp = np.clip(tmp, 0, y.max()*1.3)

        y_pred[idx] += tmp
sample_submission = pd.read_csv('/kaggle/input/ashrae/sample_submission.csv')

sample_submission['meter_reading'] = np.expm1(y_pred)

sample_submission.to_csv('submission.csv', index=False)
sns.distplot(np.log1p(sample_submission['meter_reading']))