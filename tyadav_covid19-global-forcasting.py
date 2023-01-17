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
# Import appropriate libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Read all the CSV files and create dataframes
df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv') # from Kaggle
df1 = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
df2 = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

df.head()
df2.plot()
df.plot()
plt.figure(figsize=(10,10))
sns.heatmap(df.isnull(), cbar = False, cmap = 'YlGnBu')
#setting the index to be the last_update

df.index = pd.DatetimeIndex(df.Date)
df
df['Country_Region'].value_counts()
df['Province_State'].value_counts()
# Let's plot and see the status

plt.figure(figsize = (15,10))
sns.countplot(y = 'Country_Region', data = df, order = df['Country_Region'].value_counts().iloc[:15].index)
df.resample('Y').size()
df.resample('M').size()
# Let's see the frequency

plt.plot(df.resample('M').size())
plt.title('Country Wise Per Month')
plt.xlabel('Months')
plt.ylabel('Number of confirmed cases')
# Preparing the data for forcast

df_prophet = df.resample('M').size().reset_index()
df_prophet
df_prophet.columns = ['Date', 'Confirmed_Cases']
df_prophet
df_prophet_df = pd.DataFrame(df_prophet)
df_prophet_df
# Make Predictions

df_prophet_df.columns
df_prophet_final = df_prophet_df.rename(columns={'Date': 'ds', 'Confirmed_Cases': 'y'})
df_prophet_final
m = Prophet()
m.fit(df_prophet_final)
# Forecasting future cases for 90 days

future = m.make_future_dataframe(periods = 90)
forecast = m.predict(future)
forecast
# Let's see explore forecast for next 90 days

figure = m.plot(forecast, xlabel='Date', ylabel = 'Confirmed_cases')
figure2 = m.plot_components(forecast)
