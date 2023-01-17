# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np 

import pandas as pd

# filepath: list, include the path of the csv datafile

filepath = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepath.append(os.path.join(dirname, filename))

filepath
# read the city_attributes file

for _ in filepath:

    if 'city_attributes' in _:

        city_attributes = pd.read_csv(_)

city_attributes.head(2)
# drop latitude and longitude and rename city_attributes file as city_country

to_drop = ['Latitude','Longitude']

city_country = city_attributes.drop(to_drop, axis=1)
# to_drop: list, contains cities not in US

NonUS_cities = city_country[city_country['Country'].str.contains('United')==False] #False if excluding US cities

NonUS_cities.shape  # 9 cities

to_drop = NonUS_cities.City.tolist()
# load all the weather data with only US cities only 

# unit for all the files:

# Humidity (%),Pressure (hPa), Temperature (K), Wind Direction (meteorological degrees), Wind Speed (m/s).

for _ in filepath:

    if 'temperature' in _:

        temperature = pd.read_csv(_)

        US_temperature = temperature.drop(to_drop,axis=1)

        temperature.head()

    elif 'humidity' in _:

        humidity = pd.read_csv(_)

        US_humidity = humidity.drop(to_drop,axis=1)

    elif 'pressure' in _:

        pressure = pd.read_csv(_)

        US_pressure = pressure.drop(to_drop,axis=1)

    elif 'weather_description' in _:

        weather = pd.read_csv(_)

        US_weather = weather.drop(to_drop,axis=1)

    elif 'wind_speed' in _:

        wind_speed = pd.read_csv(_)

        US_wind_speed = wind_speed.drop(to_drop,axis=1)

    elif 'wind_direction' in _:

        wind_direction = pd.read_csv(_)

        US_wind_direction = wind_direction.drop(to_drop,axis=1)
# data_w_numbers: files that the contain numbers as input

# data_w_texts: files that contain text/string as input

data_w_numbers = [US_humidity,US_temperature,US_pressure,US_wind_speed,US_wind_direction]

data_w_texts = [US_weather]

# fill NaN with mean

for data in data_w_numbers:

    data.fillna(data.mean(),inplace = True)
# convert unit for some weather files:

# original unit:

# Humidity (%),Pressure (hPa), Temperature (K), Wind Direction (meteorological degrees), Wind Speed (m/s).

# convert units to conventional units:

# Humidity (%),Pressure (atm), Temperature (oC), Wind Direction (meteorological degrees), Wind Speed (mph).

US_temperature.iloc[:,1:] =  US_temperature.iloc[:,1:].apply(lambda x:x-273.15) # celcius degree

US_pressure.iloc[:,1:] = US_pressure.iloc[:,1:].apply(lambda x:x/1013.25) # atm

US_wind_speed.iloc[:,1:] = US_wind_speed.iloc[:,1:].apply(lambda x:x*2.24) # mph
# drop nan values

US_weather2 = US_weather.dropna()

cities = US_weather2.columns.tolist()[1:]



# check all the weather conditions

# leave only one word for description -rain,smoke,clear,etc

from collections import Counter

w_uniq = [] # all the unqiue weather conditions

for col in US_weather2.columns[1:]:

    w_uniq.extend(list(set(US_weather2[col])))

w_uniq = list(set(w_uniq))

for col in US_weather2.columns[1:]:

    US_weather2[col] =  US_weather2[col].apply(lambda x:x.split()[-1])
# take an initial glance at the data

import seaborn as sns

from matplotlib import pyplot as plt

import itertools

m,n = 2,3

fig, ax = plt.subplots(m,n,figsize=(20, 10),sharey=True)

plt.suptitle('Weather condition in 27 US cities',fontsize = 24,fontweight='bold')

palette = itertools.cycle(sns.color_palette("Greens", 10))

labels = ['humidity','temperature','pressure','wind speed','wind direction']



for i,data in enumerate(data_w_numbers):

    x = data.columns[1:]

    y = data.mean()

    sns.barplot(y,x,label=labels[i],color= next(palette),orient='h',ax = ax[i//n,i%n])

    ax[i//n,i%n].set_title(labels[i])

    count = i+1

while count<m*n:

    ax[count//n,count%n].set_axis_off()

    count+=1
# We found that the pressure are pretty similiar 

# so we are only interested in humidity, temperature and wind_speed

data_w_numbers2 = [US_humidity,US_temperature,US_wind_speed]

labels = ['humidity','temperature','wind speed']

rotation = 90



palette = itertools.cycle(sns.color_palette("Greens", 10))

m,n = 1,3

fig, ax = plt.subplots(m,n,figsize=(20, 6))

plt.suptitle('Selected weather condition in 27 US cities',fontsize = 20,fontweight='bold')

for i,data in enumerate(data_w_numbers2):

    x = data.columns[1:]

    y = data.mean()

    sns.barplot(x,y,label=labels[i],color= next(palette),order = x[y.argsort()],ax = ax[i%n])

    ax[i%n].set_title(labels[i])

    ax[i%n].set_xticklabels(x[y.argsort()], rotation = rotation)
from collections import Counter

m,n = 6,5

fig, ax = plt.subplots(m,n,figsize=(20, 15))

plt.suptitle('Weather description in 27 US cities',fontsize = 24,fontweight='bold')

thres = 10

count = 0

palette = {"clear":"lemonchiffon","clouds":"lightcyan","mist":"whitesmoke","others":"lightgrey","rain":"cornflowerblue"}

# if the chance of the weather is less than thres, then we will category it into others

# this is the new dic

for col in US_weather2.columns[1:]:

    dic = dict(Counter(US_weather2[col]))

    total = sum(list(dic.values()))

    new_dic = {}

    for i in dic.keys():

        if dic[i]/total*100 < thres:

            new_dic['others'] = new_dic.get('others',0) + dic[i]

        else:

            new_dic[i] = dic[i]

    weather = list(sorted([k for k in list(new_dic.keys()) if k!='others']))+['others']

    chance = [new_dic[w] for w in weather]

    temp_ax = ax[count//n,count%n] 

    colors = [palette[w] for w in weather ]

    wedges,texts,autotexts = temp_ax.pie(chance,colors=colors,startangle=90,counterclock=False,autopct='%1.0f%%')

    temp_ax.set_title(cities[count])

    temp_ax.legend(wedges,weather,title="weather",loc="center left",bbox_to_anchor=(1,0,0.5,1))

    count+=1



while count<m*n:

    ax[count//n,count%n].set_axis_off()

    count+=1

    

plt.show()