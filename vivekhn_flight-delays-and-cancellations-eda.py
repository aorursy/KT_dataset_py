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
df_flights = pd.read_csv('../input/flight-delays/flights.csv')
df_flights.head()
# Checking the datatypes of each column

df_flights.dtypes
# Dropping unnecessary columns

df_flights = df_flights.drop(["DEPARTURE_TIME","SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL","ARRIVAL_TIME","TAIL_NUMBER","WHEELS_OFF","WHEELS_ON","TAXI_IN","TAXI_OUT","ELAPSED_TIME"],axis=1)
import matplotlib.pyplot as plt

# Counting the missing values in each variable

df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))

plt.ylabel('Percentage of missing values')

plt.xlabel('Variables')

plt.title('Quantifying missing data')

df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)

df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)

df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)

df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)

df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)
# Counting the missing values in each variable

df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))

plt.ylabel('Percentage of missing values')

plt.xlabel('Variables')

plt.title('Quantifying missing data')
df_flights['CANCELLATION_REASON'].value_counts()
df_flights['CANCELLATION_REASON'].value_counts().plot.bar(figsize=(12,6))

plt.ylabel('Number of Reasons')

plt.xlabel('Reasons')

plt.title('Listing the Missing reasons')
# Converting NaN labels to NC

df_flights['CANCELLATION_REASON'] = df_flights['CANCELLATION_REASON'].fillna('NC')

# Verifying the change

df_flights['CANCELLATION_REASON'].value_counts()
# Plotting the missing values in each variable

df_flights.isnull().mean().sort_values(ascending=False).plot.bar(figsize=(12,6))

plt.ylabel('Percentage of missing values')

plt.xlabel('Variables')

plt.axhline(y=0.02, color='red') #highlight the 2% mark with a red line:

plt.title('Quantifying missing data')
# Visualize the variable distribution with histograms

df_flights.hist(bins=30, figsize=(12,12), density=True)

plt.show()
# Determine the number of unique categories in each variable:

df_flights.nunique()
df_flights['DATE'] = pd.to_datetime(df_flights[['YEAR','MONTH', 'DAY']])
# Verifying the change

df_flights["DATE"].head()
df_airlines = pd.read_csv('../input/flight-delays/airlines.csv')

df_airlines
df_flights = df_flights.rename(columns={"AIRLINE":"IATA_CODE"})

df_merge = pd.merge(df_flights,df_airlines,on="IATA_CODE")

df_merge.head()
df_airports = pd.read_csv('../input/flight-delays/airports.csv')

df_airports = df_airports.rename(columns={"IATA_CODE":"CODE"})

df_airports
# Merging the origin details

df = df_merge.merge(df_airports[['STATE','AIRPORT','CODE']], how = 'left',

                left_on = 'ORIGIN_AIRPORT', right_on = 'CODE').drop('CODE',axis=1)

df = df.rename(columns={"STATE":"ORIGIN_STATE","AIRPORT":"ORG_AIRPORT"})

df.head()
# Merging the destination details

df = df.merge(df_airports[['STATE','AIRPORT','CODE']], how = 'left',

                left_on = 'DESTINATION_AIRPORT', right_on = 'CODE').drop('CODE',axis=1)

df = df.rename(columns={"STATE":"DESTINATION_STATE","AIRPORT":"DES_AIRPORT"})

df.head()
df.to_csv("flightdata.csv",index=False)
df["YEAR"].count()