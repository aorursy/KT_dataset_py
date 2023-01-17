import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap

%matplotlib inline 
data = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

data.head()
data.isnull().sum()
max_street_crime = data['STREET'].value_counts().index[0]

max_year_crime = data['YEAR'].value_counts().index[0]

max_hour_crime = data['HOUR'].value_counts().index[0]

max_month_crime = data['MONTH'].value_counts().index[0]

max_day_crime = data['DAY_OF_WEEK'].value_counts().index[0]



month = ['January','February','March','April','May','June','July',

         'August','September','October','November','December']



print('Street with higher occurrence of crimes:', max_street_crime)

print('Year with highest crime occurrence:', max_year_crime)

print('Hour with highest crime occurrence:', max_hour_crime)

print('Month with highest crime occurrence:', month[max_month_crime-1], max_month_crime)

print('Day with highest crime occurrence:', max_day_crime)
plt.subplots(figsize=(15,6))

sns.countplot('YEAR',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Crimes Each Year')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('HOUR',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Crimes Each Hour')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('OFFENSE_CODE_GROUP',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Types of serious crimes')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('DISTRICT',data=data,palette='RdYlGn',edgecolor=sns.color_palette('Paired',20),order=data['DISTRICT'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number Of Crimes Activities By District')

plt.show()
pd.crosstab(data.DISTRICT,data.OFFENSE_CODE_GROUP).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',5))

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
pd.crosstab(data.YEAR,data.OFFENSE_CODE_GROUP).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',5))

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
G1=data[data['MONTH'].isin(data['MONTH'].value_counts()[1:11].index)]

pd.crosstab(G1.YEAR,G1.DISTRICT).plot(color=sns.color_palette('dark',6))

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()


vand=data.loc[data.OFFENSE_CODE_GROUP=='Vandalism'][['Lat','Long']]

vand.Lat.fillna(0, inplace = True)

vand.Long.fillna(0, inplace = True) 



BostonMap=folium.Map(location=[42.356145,-71.064083],zoom_start=11)

HeatMap(data=vand, radius=16).add_to(BostonMap)



BostonMap
my = data.dropna()

df_counters = pd.DataFrame(

    {'ID' : id,

     'Name' : my.OFFENSE_CODE_GROUP,

     'lat' : my.Lat,

     'long' : my.Long,

     'region' : my.DISTRICT,

     'year': my.YEAR,

     'month': my.MONTH

    })



arrayName = []

for i in my.OFFENSE_CODE_GROUP:

    arrayName.append(i)



df_counters.head()

locations = df_counters[['lat', 'long']]

locationlist = locations.values.tolist()

BostonMap=folium.Map(location=[42.356145,-71.064083],zoom_start=11)

for point in range(0, len(locationlist)):

    string = arrayName[point]

    folium.Marker(locationlist[point], popup=string).add_to(BostonMap)

BostonMap


