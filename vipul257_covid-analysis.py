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
covid = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
covid.head(10)
covid['Date'].max()
covid.dtypes
covid['Date'] = pd.to_datetime(covid['Date'])
covid.dtypes
top = covid[covid['Date'] == covid['Date'].max()]
top.head()
top['Country/Region'].value_counts()
top_20 = top.groupby('Country/Region')['Confirmed','Deaths','Recovered','Active'].sum().sort_values('Confirmed' , ascending=False).reset_index().head(20)
top_20
top_20.plot(x='Country/Region' , y = ['Confirmed','Deaths','Recovered','Active'],
           kind = 'bar',figsize=(12,8))
US = covid[covid['Country/Region'] == 'US'][['Date','Confirmed','Deaths','Recovered','Active']]
US.plot(x = ['Date'], y = ['Confirmed','Deaths','Recovered','Active'])
US.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(US['Date'],US['Confirmed'],label='Confirmed')
plt.plot(US['Date'],US['Recovered'],label='Recovered')
plt.plot(US['Date'],US['Deaths'],label='Deaths')
plt.plot(US['Date'],US['Active'],label='Active')
plt.legend()
plt.show()
India = covid[covid['Country/Region'] == 'India'][['Date','Confirmed','Deaths','Recovered','Active']]
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(India['Date'],India['Confirmed'],label='Confirmed')
plt.plot(India['Date'],India['Recovered'],label='Recovered')
plt.plot(India['Date'],India['Deaths'],label='Deaths')
plt.plot(India['Date'],India['Active'],label='Active')
plt.legend()
plt.show()
Brazil = covid[covid['Country/Region'] == 'Brazil'][['Date','Confirmed','Deaths','Recovered','Active']]
Russia = covid[covid['Country/Region'] == 'Russia'][['Date','Confirmed','Deaths','Recovered','Active']]
SA = covid[covid['Country/Region'] == 'South Africa'][['Date','Confirmed','Deaths','Recovered','Active']]
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(India['Date'],India['Confirmed'],label='India')
plt.plot(US['Date'],US['Confirmed'],label='US')
plt.plot(Brazil['Date'],Brazil['Confirmed'],label='Brazil')
plt.plot(Russia['Date'],Russia['Confirmed'],label='Russia')
plt.plot(SA['Date'],SA['Confirmed'],label='SA')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(India['Date'],India['Active'],label='India')
plt.plot(US['Date'],US['Active'],label='US')
plt.plot(Brazil['Date'],Brazil['Active'],label='Brazil')
plt.plot(Russia['Date'],Russia['Active'],label='Russia')
plt.plot(SA['Date'],SA['Active'],label='SA')
plt.legend()
plt.show()
India
I_confirmed = India[['Date','Confirmed']]
I_active = India[['Date','Active']]
I_recovered = India[['Date','Recovered']]
I_deaths = India[['Date','Deaths']]
I_confirmed.columns=['ds','y']
I_confirmed
from fbprophet import Prophet
m1 = Prophet()
m1.fit(I_confirmed)
# Python
future = m1.make_future_dataframe(periods=10)
future.tail(15)
# Python
forecast1 = m1.predict(future)
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
I_deaths.columns=['ds','y']
m2 = Prophet()
m2.fit(I_deaths)
future = m2.make_future_dataframe(periods=10)
future.tail(15)
forecast2 = m2.predict(future)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
# Python
fig1 = m1.plot(forecast1)
# Python
fig2 = m2.plot(forecast2)
# Python
fig3 = m1.plot_components(forecast1)
fig3 = m2.plot_components(forecast2)
