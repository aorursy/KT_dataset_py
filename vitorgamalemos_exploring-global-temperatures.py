import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap

from mpl_toolkits.basemap import Basemap

from matplotlib import animation, rc

from IPython.display import HTML



%matplotlib inline 



global_temp = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv', parse_dates=['dt'])

cities_temp = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv', parse_dates=['dt'])

country_temp = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv',parse_dates=['dt'])

global_temp.head(4)
cities_temp.head(3)
country_temp.head(3)
global_temp.isnull().sum()
cities_temp.isnull().sum()
country_temp.isnull().sum()
plt.figure (figsize = (10, 9))

temperature_by_year = global_temp.groupby(global_temp.dt.dt.year).mean()

temperature_by_year.LandAverageTemperature.plot(linewidth=2, color='red',marker='x')

plt.title('Average temperature between 1750 and 2003')

plt.xlabel('year')

plt.ylabel('Temperature in Celsius')

plt.legend()

plt.grid()
plt.figure (figsize = (10, 8))

temperature_by_year = global_temp.groupby(global_temp.dt.dt.year).mean()

temperature_by_year.LandAndOceanAverageTemperature.plot(linewidth=2, color='blue',marker='o')



plt.title('Average temperature between 1750 and 2003')

plt.xlabel('Year')

plt.ylabel('Temperature in Celsius')

plt.legend()

plt.grid()
plt.figure (figsize = (10, 9))

temperature_by_year = global_temp.groupby(global_temp.dt.dt.year).mean()

temperature_by_year.LandMaxTemperature.plot(linewidth=2, color='green',marker='x')

temperature_by_year.LandMinTemperature.plot(linewidth=2, color='yellow',marker='o')

plt.title('Average temperature between 1750 and 2003')

plt.xlabel('Year')

plt.ylabel('Temperature in Celsius')

plt.legend()

plt.grid()
def getAvg(db, year, name, label):

    return db.groupby([name, year])[label].mean().unstack()



def getMin(db, year, name, label):

    return db.groupby([name, year])[label].min().unstack()



def getMax(db, year, name, label):

    return db.groupby([name, year])[label].max().unstack()



year = cities_temp.dt.dt.year

label = 'AverageTemperature'

cityAvg = getAvg(cities_temp, year, 'City', label)

cityMin = getMin(cities_temp, year, 'City', label)

cityMax = getMax(cities_temp, year, 'City', label)



fig = plt.figure(figsize=(20,10))

color = ['red','green','blue','yellow']

subplot = [221,222,223,224]

for index in range(0, 4):

    searchRandom = cityAvg.sample(10).index

    for name_city in searchRandom:

        rowAvg = cityAvg.loc[name_city] 

        rowMin = cityMin.loc[name_city] 

        rowMax = cityMax.loc[name_city] 

   

    ax1 = fig.add_subplot(subplot[index]) 

    ax1.set_title(rowMax.name)

    ax1.grid()

    ax1.plot(rowMax, label="Max")

    leg = ax1.legend()

    

    ax2 = fig.add_subplot(subplot[index]) 

    ax2.set_title(rowAvg.name)

    ax2.grid()

    ax2.plot(rowAvg, label="Avg")

    leg = ax2.legend()

    

    ax3 = fig.add_subplot(subplot[index]) 

    ax3.set_title(rowMin.name)

    ax3.grid()

    ax3.plot(rowMin, label="Min")

    leg = ax3.legend()
year = country_temp.dt.dt.year

label = 'AverageTemperature'

countryAvg = getAvg(country_temp, year, 'Country', label)

countryMin = getMin(country_temp, year, 'Country', label)

countryMax = getMax(country_temp, year, 'Country', label)



fig = plt.figure(figsize=(20,10))

subplot = [221,222,223,224]

for index in range(0, 4):

    searchRandom = countryAvg.sample(10).index

    for name_country in searchRandom:

        rowAvg = countryAvg.loc[name_country] 

        rowMin = countryMin.loc[name_country] 

        rowMax = countryMax.loc[name_country] 

   

    ax1 = fig.add_subplot(subplot[index]) 

    ax1.set_title(rowMax.name)

    ax1.grid()

    ax1.plot(rowMax, label="Max")

    leg = ax1.legend()

    

    ax2 = fig.add_subplot(subplot[index]) 

    ax2.set_title(rowAvg.name)

    ax2.grid()

    ax2.plot(rowAvg, label="Avg")

    leg = ax2.legend()

    

    ax3 = fig.add_subplot(subplot[index]) 

    ax3.set_title(rowMin.name)

    ax3.grid()

    ax3.set_xlabel('Year')

    ax3.set_ylabel('Average Temperature in °C')

    ax3.plot(rowMin, label="Min")

    leg = ax3.legend()
Russia = countryAvg.loc['Russia']

Canada = countryAvg.loc['Canada']

Greenland = countryAvg.loc['Greenland']

Finland = countryAvg.loc['Finland']



fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(221) 

ax1.plot(Russia, label="Russia")

ax1.plot(Canada, label="Canada")

ax1.plot(Finland, label="Finland")

ax1.plot(Greenland, label="Greenland")

ax1.set_xlabel('Year')

ax1.set_ylabel('Average Temperature in °C')

ax1.set_title("The average temperature of northernmost countries")

leg = ax1.legend()
Brazil = countryAvg.loc['Brazil']

Ecuador = countryAvg.loc['Ecuador']

Venezuela = countryAvg.loc['Venezuela']

Argentina = countryAvg.loc['Argentina']



fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(221) 

ax1.plot(Brazil, label="Brazil")

ax1.plot(Ecuador, label="Ecuador")

ax1.plot(Venezuela, label="Venezuela")

ax1.plot(Argentina, label="Argentina")

ax1.set_xlabel('Year')

ax1.set_ylabel('Average Temperature in °C')

leg = ax1.legend()