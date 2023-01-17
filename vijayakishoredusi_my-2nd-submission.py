
import numpy as np 
import pandas as pd 
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


Covid_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
Covid_data.head()

Covid_data['Date'] = pd.to_datetime(Covid_data['ObservationDate']).dt.normalize()
Covid_data['ObservationDate'] = Covid_data.Date.astype(str)
Covid_data = Covid_data[['ObservationDate','Province/State', 'Country/Region','Confirmed', 'Deaths', 'Recovered']]
Covid_data

january = pd.date_range('2020-01', '2020-02', freq='D')
january = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
               '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12',
               '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16',
               '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20',
               '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',
               '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28',
               '2020-01-29', '2020-01-30', '2020-01-31']

Covid_data_Jan = Covid_data[Covid_data['ObservationDate'].isin(january)]

Covid_data_Jan.head()
# 5 times it and swap the date column for dates in Auguest


df = Covid_data_Jan
df.Confirmed = df.Confirmed*5
df.Deaths = df.Deaths*5
df.Recovered = df.Recovered*5
Covid_data_Aug = df

Covid_data_Aug['ObservationDate']=Covid_data_Aug['ObservationDate'].str.replace("1","7")

print(Covid_data_Aug.head(40))

Covid_data_Aug.to_csv('/kaggle/working/my_1st_submission.csv', index=False)
