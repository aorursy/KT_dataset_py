import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/rows.csv')
df.dtypes
# check out some samples

df.head()
# how many samples are in this data set

len(df)
# convert the columns with date information into a datetime format

from datetime import datetime

df['report_dt'] = pd.to_datetime(df['Report Date Time'],format='%m/%d/%Y %I:%M:%S %p')

df['occured_dt'] = pd.to_datetime(df['Occurred Date Time'],format='%m/%d/%Y %I:%M:%S %p')

# and drop the columns that we don't need anymore

df.drop(['Report Date Time','Occurred Date Time','Report Date'

         ,'Report Time','Occurred Date','Occurred Time'],axis=1,inplace=True)
df['report_dt'].isnull().sum()
df['occured_dt'].isnull().sum()
df = df[df['report_dt'].isnull() == False]

df = df[df['occured_dt'].isnull() == False]
crimes_per_year = df['report_dt'].dt.year.value_counts().sort_index()



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

sns.set_color_codes("pastel")



g = sns.barplot(x=crimes_per_year.index, y=crimes_per_year.values,color='b')

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set(xlabel='Year', ylabel='# crimes reported')

plt.title('Number of reported crimes per year')

plt.show()
crimes_per_tod = df['occured_dt'].dt.hour.value_counts().sort_index()

g = sns.barplot(x=crimes_per_tod.index, y=crimes_per_tod.values, color='b')

g.set(xlabel='Hour', ylabel='# crimes reported')

plt.title('Crime reports per hour of day')

plt.show()
top_crimes = df['Highest Offense Description'].value_counts().head(25)

sns.set(rc={'figure.figsize':(12,8)},style="whitegrid")

g = sns.barplot(y=top_crimes.index, x=top_crimes.values,color='b')

g.set(xlabel='# crimes reported', ylabel='Offense description')

plt.title('Top 25 offenses (2003-2019)')

plt.show()
crime_coords = df[(df['Latitude'].isnull() == False) 

                  & (df['Longitude'].isnull() == False)

                 & (df['Highest Offense Description'] == 'THEFT OF BICYCLE')

                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]
import folium

from folium import plugins



map_1 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)

map_1.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))

map_1
crime_coords = df[(df['Latitude'].isnull() == False) 

                  & (df['Longitude'].isnull() == False)

                 & (df['Highest Offense Description'] == 'BURGLARY OF VEHICLE')

                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]
map_2 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)

map_2.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))

map_2
crime_coords = df[(df['Latitude'].isnull() == False) 

                  & (df['Longitude'].isnull() == False)

                 & (df['Highest Offense Description'] == 'THEFT BY SHOPLIFTING')

                 & (df['report_dt'].dt.year == 2018)][['Latitude','Longitude']]
map_3 = folium.Map(location=[30.285516,-97.736753 ],tiles='OpenStreetMap', zoom_start=11)

map_3.add_child(plugins.HeatMap(crime_coords[['Latitude', 'Longitude']].values, radius=15))

map_3