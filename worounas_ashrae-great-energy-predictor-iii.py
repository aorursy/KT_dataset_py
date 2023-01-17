# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pathlib

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype

        if col_type != 'object' and col_type != 'datetime64[ns]':

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

                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
data_dir = pathlib.Path('../input/ashrae-energy-prediction')



df_building = pd.read_csv(data_dir.joinpath('building_metadata.csv'))

df_weather_train = pd.read_csv(data_dir.joinpath('weather_train.csv'))

df_weather_test = pd.read_csv(data_dir.joinpath('weather_test.csv'))

df_train = pd.read_csv(data_dir.joinpath('train.csv'))

df_test = pd.read_csv(data_dir.joinpath('test.csv'))
df_building = reduce_mem_usage(df_building)

df_weather_train = reduce_mem_usage(df_weather_train)

df_weather_test = reduce_mem_usage(df_weather_test)

df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
print('Train Shape:', df_train.shape)

print('Test Shape:', df_test.shape)

print('Weather Train Shape:', df_weather_train.shape)

print('Weather Test Shape:', df_weather_test.shape)

print('Building Metadata Shape:', df_building.shape)
display(df_train.head())

df_train.dtypes
display(df_test.head())

df_test.dtypes
display(df_weather_train.head())

df_weather_train.dtypes
display(df_weather_test.head())

df_weather_test.dtypes
display(df_building.head())

df_building.dtypes
df_train = df_train.merge(df_building, on='building_id', how='left')

df_train = df_train.merge(df_weather_train, on=['site_id', 'timestamp'], how='left')



df_test = df_test.merge(df_building, on='building_id', how='left')

df_test = df_test.merge(df_weather_test, on=['site_id', 'timestamp'], how='left')



del df_building, df_weather_train, df_weather_test
print('Train Shape:', df_train.shape)

df_train.head()
print('Test Shape:', df_test.shape)

df_test.head()