import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import random

import seaborn as sns

from fbprophet import Prophet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# dataframes creation for both training and testing datasets 

chicago_df_1 = pd.read_csv('/kaggle/input/Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)

chicago_df_2 = pd.read_csv('/kaggle/input/Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)

chicago_df_3 = pd.read_csv('/kaggle/input/Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)

print(chicago_df_1.shape, chicago_df_2.shape, chicago_df_3.shape)
chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False, axis=0)
chicago_df.head()
# Let's see how many null elements are contained in the data

plt.figure(figsize=(10,10))

sns.heatmap(chicago_df.isnull(), cbar = False, cmap = 'YlGnBu')
chicago_df.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)
# Assembling a datetime by rearranging the dataframe column "Date". 



chicago_df.Date = pd.to_datetime(chicago_df.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date 

chicago_df.index = pd.DatetimeIndex(chicago_df.Date)
chicago_df['Primary Type'].value_counts()
plt.figure(figsize = (15, 10))

sns.countplot(y= 'Primary Type', data = chicago_df, order = chicago_df['Primary Type'].value_counts().iloc[:15].index)
plt.figure(figsize = (15, 10))

sns.countplot(y= 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().iloc[:15].index)
chicago_df
# Resample is a Convenience method for frequency conversion and resampling of time series.



plt.plot(chicago_df.resample('Y').size())

plt.title('Crimes Count Per Year')

plt.xlabel('Years')

plt.ylabel('Number of Crimes')
# Resample is a Convenience method for frequency conversion and resampling of time series.



plt.plot(chicago_df.resample('M').size())

plt.title('Crimes Count Per Month')

plt.xlabel('Months')

plt.ylabel('Number of Crimes')
plt.plot(chicago_df.resample('Q').size())

plt.title('Crimes Count Per Quarter')

plt.xlabel('Quarters')

plt.ylabel('Number of Crimes')
chicago_prophet = chicago_df.resample('M').size().reset_index()

chicago_prophet.columns = ['Date', 'Crime Count']

chicago_prophet_df = pd.DataFrame(chicago_prophet)

chicago_prophet
chicago_prophet_df_final = chicago_prophet_df.rename(columns={'Date':'ds', 'Crime Count':'y'})

chicago_prophet_df_final
m = Prophet()

m.fit(chicago_prophet_df_final)



# Forcasting into the future

future = m.make_future_dataframe(periods=1825)

forecast = m.predict(future)
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
figure3 = m.plot_components(forecast)