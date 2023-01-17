import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap

%matplotlib inline 



data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='latin-1')

data.head()
data.columns
data.isnull().sum()
attack=data.loc[data.iyear==2002][['latitude','longitude']]

attack.latitude.fillna(0, inplace = True)

attack.longitude.fillna(0, inplace = True) 



World =folium.Map(location=[0,0],zoom_start=2)

HeatMap(data=attack, radius=16).add_to(World)



print('Terrorism around the world in 2002')

World
#my = data.dropna()



my = data

df_counters = pd.DataFrame(

    {'ID' : id,

     'lat' : my.latitude,

     'long' : my.longitude,

     'region' : my.country_txt,

     'year': my.iyear,

     'type': my.attacktype1_txt

    })



df_counters = df_counters.dropna()

arrayName = []

arrayCountry = []

arrayYear = []



for i in df_counters['region']:

    arrayCountry.append(i)

    

for i in df_counters['year']:

    arrayYear.append(i)

    

for i in range(len(df_counters)):

    arrayName.append(i)

    

df_counters.head()

locations = df_counters[['lat', 'long']]

locationlist = locations.values.tolist()

BostonMap=folium.Map(location=[42.738006, -123.417103],zoom_start=4)

for point in range(0, len(locationlist)):

    string = arrayYear[point]

    if arrayCountry[point] == 'United States' and string >= 2001:

        folium.Marker(locationlist[point], popup=string).add_to(BostonMap)



print('Terrorist attacks in the United States after 2001 - Location')

BostonMap
df_counters.head()

arrayType = []

for i in df_counters['type']:

    arrayType.append(i)

    

locations = df_counters[['lat', 'long']]

locationlist = locations.values.tolist()

BostonMap = folium.Map(location=[42.738006, -123.417103],zoom_start=4)

color = 'blue'

for point in range(0, len(locationlist)):

    string = arrayYear[point]

    if arrayCountry[point] == 'United States' and string >= 1980:

        if arrayType[point] in 'Bombing/Explosion':

            color = 'red'

        elif arrayType[point] in 'Assassination':

            color = 'green'

        elif arrayType[point] in 'Armed Assault':

            color = 'purple'

        else:

            color = 'blue'

              

        typeFormat = '{} - {}'.format(arrayType[point], string)

        folium.Marker(locationlist[point], 

                      popup=typeFormat,

                      icon=folium.Icon(color=color)).add_to(BostonMap)

        

print('Terrorist attacks in the United States after 1980 - Location and TypeAttack')

BostonMap
plt.subplots(figsize=(15,6))

sns.countplot('iyear',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Terrorist attacks between 1970 and 2017')

plt.plot(color="white", lw=2)

plt.show()
data['attacktype1_txt']
data['attacktype1_txt'] == 'Assassination'



countAssassination = 0

countBombing = 0

countArmedAssault = 0

for nameAttack in data['attacktype1_txt']:

    if nameAttack in 'Assassination':

        countAssassination += 1

    elif nameAttack in 'Bombing/Explosion':

        countBombing += 1

    elif nameAttack in 'Armed Assault':

        countArmedAssault += 1

        

arrayType = ['Assassination','Bombing','ArmedAssault']

arrayCount = [countAssassination, countBombing, countArmedAssault]



plt.bar(arrayType, arrayCount)

plt.title('TypeArrack: Assassination X Bombing X ArmedAssault')

df = pd.DataFrame({'TypeArrack Pie Plot': arrayCount},index=arrayType)

plot = df.plot.pie(y='TypeArrack Pie Plot', figsize=(5, 5))
plt.subplots(figsize=(13,6))

sns.countplot(y='attacktype1_txt',data=data)

plt.title('Favorite Attack')

plt.show()
plt.subplots(figsize=(13,6))

sns.countplot(y='region_txt',data=data)

plt.title('Number of attacks per region')

plt.show()
plt.subplots(figsize=(13,45))

sns.countplot(y='country_txt',data=data)

plt.title('Number of attacks per country')

plt.show()
countRegionAttack = data['region_txt'].value_counts().to_frame()

countRegionAttack.columns=['Attacks']



countRegionKill = data.groupby('region_txt')['nkill'].sum().to_frame()

countRegionAttack.merge(countRegionKill,

                        left_index=True, 

                        right_index=True).plot.bar()

plt.show()