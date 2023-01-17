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

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

import datetime as dt

import seaborn as sns

%matplotlib inline
trend = pd.read_csv('../input/coronavirusdataset/trend.csv',parse_dates=['date'])

route = pd.read_csv('../input/coronavirusdataset/route.csv',parse_dates=['date'])

patient = pd.read_csv('../input/coronavirusdataset/patient.csv',parse_dates=['confirmed_date'])

time=pd.read_csv('../input/coronavirusdataset/time.csv',parse_dates=['date'])
plt.figure(figsize=(12,8))

plt.plot(time['date'],time['new_confirmed'],'ro:')

# sns.lineplot(data=time,x='date',y='new_confirmed',markers=True,dashes=True)

plt.xticks(rotation=30)

plt.title('How many COVID-19 confirmed peple')
fig,ax1=plt.subplots(figsize=(12,8))

ax2 = ax1.twinx()

confirmed_pivot = time.pivot_table(index='date',values='new_confirmed',aggfunc='sum').reset_index()

sns.lineplot(x=confirmed_pivot['date'],y=confirmed_pivot['new_confirmed'].cumsum(),ax=ax1,label='confirmed')

ax1.set_ylim=[0,10000]

ax1.set_ylabel('Confirmation Count')

ax1.legend(loc='upper left')

ax1.tick_params(rotation=60)



deceased_pivot = time.pivot_table(index='date',values='new_deceased',aggfunc='sum').reset_index()

sns.lineplot(x=deceased_pivot['date'],y=deceased_pivot['new_deceased'].cumsum(),color='blue',ax=ax2,label='deceased')

ax2.set_ylabel('Decease Count')

ax2.set_ylim=[0,100]

ax2.legend(loc='upper center')

ax2.tick_params(rotation=60)





plt.title('Confirmation and Decease Accumulate Count')

plt.show()
plt.figure(figsize=(12,8))

released_pivot = time.pivot_table(index='date',values='new_released',aggfunc='sum').reset_index()

sns.lineplot(x=released_pivot['date'],y=released_pivot['new_released'].cumsum(),label='released')

deceased_pivot = time.pivot_table(index='date',values='new_deceased',aggfunc='sum').reset_index()

sns.lineplot(x=deceased_pivot['date'],y=deceased_pivot['new_deceased'].cumsum(),color='blue',label='deceased')

plt.xticks(rotation=30)

plt.legend(loc=0)

plt.title('Released and Deceased accumulate Count')
fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,8))

sns.countplot(patient['group'],ax=ax1)

ax1.tick_params(labelrotation=90)

ax1.set(title='Belonging to the infected')

sns.countplot(patient['infection_reason'],ax=ax2)

ax2.tick_params(labelrotation=90)

ax2.set(title='Infection Reason')
idx= patient[patient['group'] == 'Shincheonji Church']['infection_reason'].isnull().index
patient['infection_reason'][idx] = 'visit to Shincheonji Church'
sns.countplot(patient['infection_reason'])

plt.xticks(rotation=90)
plt.figure(figsize=(12,8))

sns.countplot(patient['region'])

plt.xticks(rotation=90)

plt.title('Where is most infected')
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(route['latitude'], route['longitude'],route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
import plotly.express as px

states = pd.DataFrame(patient["state"].value_counts())

states["status"] = states.index

states.rename(columns={"state": "count"}, inplace=True)



fig = px.pie(states,

             values="count",

             names="status",

             title="Current state of patients",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")

fig.show()

# Thanks for Vansh Jatana
col = ["confirmed_date", "released_date", "deceased_date"]

for i in col:

    patient[i] = pd.to_datetime(patient[i])

patient['country'].fillna(patient['country'].mode()[0],inplace=True)

patient['confirmed_date'].fillna(dt.datetime(2020,2,2),inplace=True)

patient['state'].fillna(patient['state'].mode()[0],inplace=True)
for i in range(patient.shape[0]):

    if type(patient['deceased_date'][i]) == float:

        if patient['state'][i] == 'released':

            patient['deceased_date'][i] = 'survived'

        elif patient['state'][i] == 'isolated':

            patient['deceased_date'][i] = "don't know"

    else:

        continue

for i in range(patient.shape[0]):

    if type(patient['released_date'][i]) == float:

        if patient['state'][i] == 'deceased':

            patient['released_date'][i] = 'deceased'

        else:

            patient['released_date'][i] = "don't know"

    else:

        continue
idx = patient[patient['group']=='Shincheonji Church'].index

patient['infected_by'][idx] = 31.0

patient['group'].fillna('None',inplace=True)

patient['infection_order'] = patient['infection_order'].fillna(0.0).astype(int)
patient['birth_year'] = patient['birth_year'].fillna(0.0).astype(int)

patient['birth_year'] = patient['birth_year'].apply(lambda x: x if x>0 else np.nan)

patient['age'] = 2020-patient['birth_year'] + 1 # korean age

import math

def age_range(age):

    if age >= 0: # not NaN

        if age % 10 != 0:

            lower = int(math.floor(age / 10.0)) * 10

            upper = int(math.ceil(age / 10.0)) * 10 - 1

            return f"{lower}-{upper}"

        else:

            lower = int(age)

            upper = int(age + 9) 

            return f"{lower}-{upper}"

    return "Unknown"

patient['age_range'] = patient['age'].apply(lambda x: age_range(x))
patient['relased_time'] = patient['released_date'] - patient['confirmed_date']

patient['deceased_time'] = patient['deceased_date'] - patient['confirmed_date']
tmp=[]

idx = []

for i in range(patient.shape[0]):

    if type(patient['sex'][i]) == str:

        tmp.append(patient['age_range'][i]+'_'+patient['sex'][i])

        idx.append(i)

    else:

        continue

patient['age_sex']='None'

patient['age_sex'][idx] = tmp
patient.tail()
released = patient[patient['state'] == 'released']

fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,5))

sns.countplot(released['age_range'].sort_values(),ax=ax1)

sns.kdeplot(released['age'],shade=True,ax=ax2)
tmp = pd.DataFrame(released['age_range'].value_counts()/patient['age_range'].value_counts())

tmp = tmp.dropna()

tmp.plot(kind='bar')
deceased = patient[patient['state'] == 'deceased']

fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,5))

sns.countplot(deceased['age_range'].sort_values(),ax=ax1)

ax1.set(title='Deceased')

sns.kdeplot(deceased['age'],shade=True,ax=ax2)
tmp = pd.DataFrame(deceased['age_range'].value_counts()/patient['age_range'].value_counts())

tmp = tmp.dropna()

tmp.plot(kind='bar')
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(20,8))

sns.countplot(deceased['age_sex'].sort_values(ascending=True),ax=ax1)

ax1.tick_params(rotation=90)

sns.countplot(deceased['age_range'],hue=deceased['sex'],ax=ax2)

sns.countplot(deceased['sex'],ax=ax3)
sns.countplot(deceased['region'])
tmp=time['new_confirmed'].cumsum()

dataset = pd.concat([time['date'],tmp],axis=1)

dataset = dataset[30:]
from fbprophet import Prophet
prophet = pd.DataFrame(dataset)

prophet.columns = ['ds','y']

prophet
m=Prophet()

m.fit(prophet)
future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)

forecast.tail(10)
from fbprophet.plot import plot_plotly

import plotly.offline as py

fig = plot_plotly(m, forecast)

py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
time.head()
daily_count = patient.groupby(patient.confirmed_date).id.count()

dataset = daily_count.resample('D').first().fillna(0).cumsum()

dataset = dataset[30:]
tmp=time['new_deceased'].cumsum()

deceased_prophet = pd.concat([time['date'],tmp],axis=1)

deceased_prophet.columns = ['ds','y']

deceased_prophet = deceased_prophet[35:]

deceased_prophet
n=Prophet()

n.fit(deceased_prophet)

future_deceased = n.make_future_dataframe(periods=10)

forecast_deceased = n.predict(future_deceased)

forecast_deceased.tail(10)
fig = plot_plotly(n, forecast_deceased)

py.iplot(fig) 

fig = n.plot(forecast_deceased,xlabel='Date',ylabel='Deceased Count')
tmp=time['new_released'].cumsum()

prophet_released = pd.concat([time['date'],tmp],axis=1)

prophet_released.columns=['ds','y']

prophet_released = prophet_released[40:]

prophet_released
j=Prophet()

j.fit(prophet_released)

future_released = j.make_future_dataframe(periods=10)

forecast_released = j.predict(future_released)

forecast_released.tail(10)
fig = plot_plotly(j, forecast_released)

py.iplot(fig) 

fig = j.plot(forecast_released,xlabel='Date',ylabel='Released Count')
confirmed_forecast = forecast[['ds','trend']].tail(10)

confirmed_forecast.columns=['date','predict_confirmed']

released_forecast = forecast_released[['ds','trend']].tail(10)

released_forecast.columns=['date','predict_released']

released_forecast['date'] = released_forecast['date'].astype(str)

deceased_forecast = forecast_deceased[['ds','trend']].tail(10)

deceased_forecast.columns = ['date','predict_deceased']

for i in [confirmed_forecast,released_forecast,deceased_forecast]:

    i['date'] = i['date'].astype(str)

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(20,12))

sns.pointplot(data=confirmed_forecast,x='date',y='predict_confirmed',color='purple',ax=ax1)

ax1.tick_params(rotation=60)

ax1.set(title='Predict confirmer 3.10~19')



sns.pointplot(data=released_forecast,x='date',y='predict_released',color='blue',ax=ax2)

ax2.tick_params(rotation=60)

ax2.set(title='Predict to be released 3.10~19')



sns.pointplot(data=deceased_forecast,x='date',y='predict_deceased',ax=ax3)

ax3.tick_params(rotation=60)

ax3.set(title='Predict dead 3.10~19')



plt.legend(loc=0)
for i in [confirmed_forecast,released_forecast,deceased_forecast]:

    print(i.tail(1))