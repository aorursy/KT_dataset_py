# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nCoV_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

nCoV_df.head()
# Importing the world_coordinates dataset

world_coordinates = pd.read_csv('../input/world-coordinates/world_coordinates.csv')

world_coordinates.head()
country_level = nCoV_df.groupby("Country").agg({

    "Confirmed": max, 

    "Deaths": max, 

    "Recovered": max

}).reset_index()



country_level.head()
fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(111)

x = np.array([i for i in range(len(country_level['Country']))])



bar_widh = 0.3

ax.bar(x, height= country_level['Confirmed'], color = 'b', tick_label  = country_level['Country'], width = bar_widh, label="Confirmed") 

ax.bar(x+bar_widh, height= country_level['Deaths'], color = 'r', width = bar_widh, label="Deaths") 

ax.bar(x+2*bar_widh, height= country_level['Recovered'], color = 'g', width = bar_widh, label="Recovered") 

ax.set_xticklabels(country_level['Country'], rotation=70)



ax.yaxis.set_major_locator(plt.MultipleLocator(500))

for i,v in enumerate(country_level['Confirmed']):

    ax.text(x[i] - bar_widh/2, v, str(v), color = 'b')

    

for i,v in enumerate(country_level['Deaths']):

    if v>0:

        ax.text(x[i] , v + 100, str(v), color = 'r')

    

for i,v in enumerate(country_level['Recovered']):

    if v >0:

        ax.text(x[i] + bar_widh/2, v + 300, str(v), color = 'g')

    

ax.legend()

plt.show()
covid19_world_coordinates = pd.merge(country_level, world_coordinates, on="Country", how="left")

covid19_world_coordinates.head()
# covid19_world_coordinates[pd.isna(covid19_world_coordinates.latitude) | pd.isna(covid19_world_coordinates.longitude)]

china = covid19_world_coordinates[covid19_world_coordinates.Country =='China']

mainland_china = covid19_world_coordinates[covid19_world_coordinates.Country =='Mainland China']

covid19_world_coordinates = covid19_world_coordinates[~covid19_world_coordinates.Country.isin(['China', 'Mainland China'])]
china_df = pd.concat([china, mainland_china], axis=0)

china = (china_df

         .fillna(0)

         .sum()

)



china = china.to_frame().transpose()

china['Country'] = 'China'

china['Code'] = 'CN'



china
covid19_world_coordinates = pd.concat([covid19_world_coordinates, china])
covid19_world_coordinates[covid19_world_coordinates.Country=='China']
covid19_world_coordinates.dropna(inplace=True)
import folium 



# create map and display it

world_map = folium.Map(location=[10, -20], zoom_start=2.3)



for lat, lon, value, name in zip(covid19_world_coordinates['latitude'], covid19_world_coordinates['longitude'],\

                                 covid19_world_coordinates['Confirmed'], covid19_world_coordinates['Country']):

    folium.CircleMarker([lat, lon],

                        radius=min(100, value),

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
mainland_china_level = nCoV_df[nCoV_df.Country=='Mainland China'].groupby("Province/State").agg({

    "Confirmed": max, 

    "Deaths": max, 

    "Recovered": max

}).reset_index()



mainland_china_level.head()
fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(111)

x = np.array([i for i in range(len(mainland_china_level['Province/State']))])



bar_widh = 0.3

ax.bar(x, height= mainland_china_level['Confirmed'], color = 'b', tick_label  = mainland_china_level['Province/State'], width = bar_widh, label="Confirmed") 

ax.bar(x+bar_widh, height= mainland_china_level['Deaths'], color = 'r', width = bar_widh, label="Deaths") 

ax.bar(x+2*bar_widh, height= mainland_china_level['Recovered'], color = 'g', width = bar_widh, label="Recovered") 

ax.set_xticklabels(mainland_china_level['Province/State'], rotation=70)



ax.yaxis.set_major_locator(plt.MultipleLocator(500))

for i,v in enumerate(mainland_china_level['Confirmed']):

    ax.text(x[i] - bar_widh/2, v, str(v), color = 'b')

    

for i,v in enumerate(mainland_china_level['Deaths']):

    if v>0:

        ax.text(x[i] , v + 100, str(v), color = 'r')

    

for i,v in enumerate(mainland_china_level['Recovered']):

    if v >0:

        ax.text(x[i] + bar_widh/2, v + 300, str(v), color = 'g')

    

ax.legend()

ax.grid()

plt.show()
world_level_df = nCoV_df.groupby(['Date']).agg({

    'Confirmed': sum,

    'Deaths': sum, 

    'Recovered': sum

}).reset_index()

world_level_df.head()
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111)

ax.plot(world_level_df['Date'], world_level_df['Confirmed'])

ax.plot(world_level_df['Date'], world_level_df['Deaths'])

ax.plot(world_level_df['Date'], world_level_df['Recovered'])

ax.set_xticklabels(world_level_df['Date'], rotation = 70)

ax.legend()

plt.show()
world_level_df['lag_confirmed'] = world_level_df['Confirmed'].shift(1)

world_level_df['lag_deaths'] = world_level_df['Deaths'].shift(1)

world_level_df['lag_recovered'] = world_level_df['Recovered'].shift(1)



world_level_df['movement_confirmed'] = world_level_df['Confirmed'] - world_level_df['lag_confirmed']

world_level_df['movement_deaths'] = world_level_df['Deaths'] - world_level_df['lag_deaths']

world_level_df['movement_recovered'] = world_level_df['Recovered'] - world_level_df['lag_recovered']



world_level_df.head()
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111)

ax.plot(world_level_df['Date'], world_level_df['movement_confirmed'])

ax.plot(world_level_df['Date'], world_level_df['movement_deaths'])

ax.plot(world_level_df['Date'], world_level_df['movement_recovered'])

ax.set_xticklabels(world_level_df['Date'], rotation = 70)

ax.legend()

plt.show()
sars = pd.read_excel("../input/sars-who-data/sars_final.xlsx")

ebola = pd.read_csv("../input/ebola-cases/ebola.csv")

ebola.head()
sars.head()
ebola = ebola[ebola.Indicator=="Cumulative number of confirmed Ebola cases"].groupby("Date").agg({

    "value": sum

}).reset_index()
ebola.head()
print(world_level_df.shape)

print(sars.shape)

print(ebola.shape)
m = world_level_df.shape[0]

sars = sars.loc[0:m-1, :]

ebola = ebola.loc[0:m-1, :]
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111)

x = [i for i in range(m)]

ax.plot(x, world_level_df['Confirmed'], color='b', label = "nCov")

ax.plot(x, sars['Infected'], color="r", label = "Sars")

ax.plot(x, ebola['value'], color='yellow', label = "Ebola")



ax.legend()

ax.title.set_text('Effected numbers in first {0} of nCov, Sars and Ebola'.format(m))

plt.show()