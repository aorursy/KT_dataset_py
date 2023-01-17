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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import geopandas as gpd

import squarify    # pip install squarify (algorithm for treemap)

from mpl_toolkits.basemap import Basemap

import folium

from folium.plugins import HeatMap
df = pd.read_csv('../input/california-housing-prices/housing.csv')

df.head()
df.ocean_proximity.value_counts()
plt.figure(figsize=(16,7))

df['ocean_proximity'].value_counts().plot(kind='bar',edgecolor='k', alpha=0.8)

  

for index, value in enumerate(df['ocean_proximity'].value_counts()):

    plt.text(index, value, str(value))

plt.xlabel("Area", fontsize=14)

plt.ylabel("Houses", fontsize=13)

plt.xticks(rotation=0)

plt.title("Houses across the state of California (CA)", fontsize=15)

plt.show() 
california_img = plt.imread('../input/ca-stuff/ca_map.png')



df.plot(kind='scatter', x='longitude', y='latitude',  alpha=0.3,

                s=df['population']/50, label='population', figsize=(10,7), c='b', zorder=2)



plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=1)

plt.title('Population across California',fontsize=15)

plt.legend() 

plt.show()
california_img = plt.imread('../input/ca-stuff/ca_map.png')



df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,

        s=df['population']/100, label='population', figsize=(10,7),

        c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)



plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=1)

plt.title('Median House Value across California',fontsize=15)

plt.legend() 

plt.show()
df.median_house_value.head()
plt.figure(figsize=(16,7))



df['median_house_value'].hist(bins=100)

plt.xlabel("Median House Value", fontsize=14)

plt.ylabel("Houses", fontsize=13)

plt.xticks(rotation=0)

plt.title("Median House Value across the state of California (CA)", fontsize=15)

plt.show() 
plt.figure(figsize=(20,5))

sns.set_color_codes(palette="bright")

sns.distplot(df['median_house_value'],color='g')

plt.title("Median House Value across the state of California (CA)", fontsize=15)

plt.xlabel("Median House Value", fontsize=13)

plt.ylabel("Houses", fontsize=13)

plt.show()
plt.figure(figsize=(12,8))

sns.jointplot(x=df.latitude.values,y=df.longitude.values,height=10, alpha=0.5)

plt.ylabel("longitude")

plt.xlabel("latitude")

plt.show()
df.columns
# Create the GeoDataFrame

df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]) )



df.head()
california = gpd.read_file('../input/ca-stuff/f067d2f7-7950-4e16-beba-8972d811599c2020329-1-18infjv.25og.shp')

california.head()
# Using groupby to plot World Heritage Sites that are in the danger list

ax=california.plot(figsize=(15,10),linestyle=":",edgecolor='black',color='lightgray')

df.groupby("ocean_proximity")['geometry'].plot(ax=ax,marker='o', markersize=50, alpha=0.4)

plt.title("Houses by Ocean Proximity across California", fontsize=14)

plt.show()
m = folium.Map(location=[37.166773, -120.436393], tiles="OpenStreetMap", zoom_start=6)

 

# I can add marker one by one on the map

for i in range(0,len(df)):

    folium.Circle([df.iloc[i]['latitude'], df.iloc[i]['longitude']], popup=df.iloc[i]['ocean_proximity'], radius=10).add_to(m)

#m.save('Folium CA.png')  <-----To download the graph

m
map_ca = folium.Map(location=[36.7783,-119.4179],

                    zoom_start = 6, min_zoom=5) 



df1 = df[['latitude', 'longitude']]

data = [[row['latitude'],row['longitude']] for index, row in df1.iterrows()]

HeatMap(data, radius=10).add_to(map_ca)

map_ca