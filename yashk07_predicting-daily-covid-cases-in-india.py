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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from fbprophet import Prophet
covid_df = pd.read_csv('../input/daily-covid-cases-india/datasets_549966_1364659_nation_level_daily.csv')
covid_df.head()
covid_df.info()
covid_df.Date

covid_df['Date'] = '2019 ' + covid_df['Date'].astype('str')
covid_df.Date
covid_df['Date'] = covid_df['Date'].str.replace("January","01")

covid_df['Date'] = covid_df['Date'].str.replace("February","02")

covid_df['Date'] = covid_df['Date'].str.replace("March","03")

covid_df['Date'] = covid_df['Date'].str.replace("April","04")

covid_df['Date'] = covid_df['Date'].str.replace("May","05")

covid_df['Date'] = covid_df['Date'].str.replace("June","06")

covid_df['Date'] = covid_df['Date'].str.replace("July","07")
covid_df['Date'] = covid_df['Date'].apply(lambda x : x.split(" ")[0] +"-"+ x.split(" ")[2] + "-" + x.split(" ")[1])
covid_df.Date
prophet_covid_df1 = covid_df[['Date','Daily Confirmed']]

prophet_covid_df2 = covid_df[['Date','Daily Recovered']]
prophet_covid_df1.tail()
prophet_covid_df1 = prophet_covid_df1.rename(columns={'Date':'ds','Daily Confirmed' : 'y'})
prophet_covid_df1.head()
prophet_covid_df1 = prophet_covid_df1.drop([30]) #30th row gave a parse error
pr = Prophet()
pr.fit(prophet_covid_df1)
future = pr.make_future_dataframe(periods = 153)#For the next 153 days of the dataset
forecast = pr.predict(future)
forecast[forecast['ds']=='2019-09-15']
figure = pr.plot(forecast)

plt.xlabel('Date')

plt.ylabel('Confirmed COVID cases')
figure = pr.plot_components(forecast)