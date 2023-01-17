# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings(action = 'ignore')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/nyc_taxi_trip_duration.csv')

data.head()
data.columns
data.shape
data.isna().sum()
data.dtypes
data.dtypes[data.dtypes == 'int64']
data.dtypes[data.dtypes == 'float64']
data.dtypes[data.dtypes == 'object']
data[['id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag']].head()
data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype('category')
data.dtypes
pickup_date = pd.DatetimeIndex(data['pickup_datetime'])

dropoff_date = pd.DatetimeIndex(data['dropoff_datetime'])
data['pickup_hour'] = pickup_date.hour

data['dropoff_hour'] = dropoff_date.hour

data['pickup_dow'] = pickup_date.dayofweek

data['dropoff_dow'] = dropoff_date.dayofweek

# data['pickup_date'] = pickup_date.date

# data['dropoff_date'] = dropoff_date.date
data.drop(columns=['pickup_datetime','dropoff_datetime'], inplace=True)
data.dtypes
data.describe()
data.columns
numerical_col = ['vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude','trip_duration', 'pickup_hour', 'dropoff_hour',

       'pickup_dow', 'dropoff_dow']
def univariate_analysis(data, column_group):

    size = len(column_group)

    plt.figure(figsize=(3*size, 20), dpi=100)

    for i, col in enumerate(column_group):

        maxi = data[col].max()

        mini = data[col].min()

        st_dev = data[col].std()

        mean = data[col].mean()

        median = data[col].median()

        ran = maxi - mini

        skew = data[col].skew()

        kurt = data[col].kurtosis()  

        points = mean-st_dev, mean+st_dev

        

        plt.subplot(4,3, i+1)

        plt.subplots_adjust(left=None, bottom=2, right=None, top=3,

                wspace=0.4, hspace=0.6)

        

        sns.kdeplot(data[col], shade=True)

        plt.xlabel('{}'.format(col), fontsize = 20)

        plt.ylabel('density')

        plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),

                                                                                                   round(kurt,2),

                                                                                                   round(skew,2),

                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),

                                                                                                   round(mean,2),

                                                                                                   round(median,2)), fontsize=20)

        

        
univariate_analysis(data, numerical_col)