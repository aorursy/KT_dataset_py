# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input/ashrae-energy-prediction/"))
root = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
import numpy as np

import pandas as pd

import os

import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.patches as patches



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 100)
train_df = pd.read_csv(root + "train.csv")

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')

train_df.head()
print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
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
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)
train_df.head()
plt.figure(figsize = (15,5))

train_df['meter_reading'].plot()
from statsmodels.tsa.seasonal import seasonal_decompose



building_id = 200

building = train_df.query(f'building_id == {building_id} and meter==0')

building.head()
train_df.building_id.nunique()

building['meter_reading'].hist(figsize=(6, 5))
energy_types = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}



plt.figure(figsize=(16,5))

train_df.meter.value_counts().plot(kind="bar")

plt.title(f"Most readings measure electricity")

plt.ylabel("Count of measurements")

plt.xlabel(f"Type of meter")

plt.xticks(range(0,4), list(energy_types.values()), rotation=45)

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

train_df.groupby("meter").meter_reading.sum().plot(kind="bar")

plt.title(f"Most consumed energy is for generating steam")

plt.ylabel("Sum of consumed energy")

plt.xlabel(f"Type of energy")

plt.xticks(range(0,4), list(energy_types.values()), rotation=45)

plt.tight_layout()

plt.show()
def plot_dist_col(column):

    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.distplot(weather_train_df[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)

    sns.distplot(weather_test_df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['train', 'test'])

    plt.show()
plot_dist_col('air_temperature')