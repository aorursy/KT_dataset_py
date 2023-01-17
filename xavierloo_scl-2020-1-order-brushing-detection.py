# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/order_brush_order.csv')

df.head()
df.index
df.columns
df.info()

df.describe()
sum(df.duplicated())
# Take a sample shop for the test

shopid = df['shopid'][0]

shopid
shop_1 = df.loc[ df['shopid'] == 93950878 ]

shop_1
# Sort by event_time in ascending order

shop_1 = shop_1.sort_values(by=['event_time'])

shop_1
(shop_1['event_time'].min(), shop_1['event_time'].max())
from datetime import datetime, date



# Convert date string to datetime object

# Ref: https://www.programiz.com/python-programming/datetime/strptime

#

# %Y - Year in four digits.

# %m - Month as a zero-padded decimal number.

# %d - Day of the month as a zero-padded decimal.

# 

# %H - Hour (24-hour clock) as a zero-padded decimal number.

# %M - Minute as a zero-padded decimal number.

# %S - Second as a zero-padded decimal number.

time_1 = datetime.strptime(shop_1['event_time'].iloc[0], "%Y-%m-%d %H:%M:%S")

time_2 = datetime.strptime(shop_1['event_time'].iloc[1], "%Y-%m-%d %H:%M:%S")



(time_1, time_2)
# Difference between two dates and times

time_2 - time_1
# Try to resampling to 1 hour

# Ref:

#      - https://riptutorial.com/pandas/example/7083/downsampling-and-upsampling

#      - https://machinelearningmastery.com/resample-interpolate-time-series-data-python/

# NOTE: Resampling issue cannot resolve

shop_1.resample('60min')
prev_time = ''



for curr_time in shop_1['event_time']:

    if (prev_time == ''):

        prev_time = curr_time

        continue

        

    t1 = datetime.strptime(prev_time, "%Y-%m-%d %H:%M:%S")

    t2 = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S")

    

    print(t2 - t1)