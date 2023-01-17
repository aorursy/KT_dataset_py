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

from matplotlib import rcParams

import matplotlib

import seaborn as sns

import geopandas as gpd

from geopandas.tools import geocode

import squarify    # pip install squarify (algorithm for treemap)

from mpl_toolkits.basemap import Basemap

import folium

import plotly.express as px

import plotly.graph_objects as go
df= gpd.read_file('../input/protected-shipwrecks-sites/ProtectedWrecks_17July2020.shp')

df.head(2)
data = df.loc[:, ["Name", "DesigDate", "Latitude","Longitude","AREA_HA","geometry"]].copy()

data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

data.head(2)
data.crs = {'init': 'epsg:4326'}

print(data.crs)
gb = gpd.read_file("../input/uk-poly/Uk_poly.shp")

gb.head()
ax = gb.plot(figsize=(10,8), color='lightgray', linestyle=':', edgecolor='black')

data.plot(markersize=70,alpha=0.9,edgecolor='k',ax=ax,color='lime')

plt.title("Protected Shipwrecks Sites in English territorial waters")

plt.show()
gb1 = gpd.read_file("../input/uk-poly/England_AL4-AL4.shp")

gb1.head()
ax = gb1.plot(figsize=(10,8), color='lightgray', linestyle=':', edgecolor='black')

data.plot(markersize=70,alpha=0.9,edgecolor='k',ax=ax,color='lime')

plt.title("Protected Shipwrecks Sites in English territorial waters")

plt.show()
m = folium.Map(location=[52.007666, -1.241896], tiles="OpenStreetMap", zoom_start=6.4)

 

# I can add marker one by one on the map

for i in range(0,len(data)):

    folium.Marker([data.iloc[i]['Latitude'], data.iloc[i]['Longitude']], popup=data.iloc[i]['Name']).add_to(m)



# m.save('Folium Shipwrecks sites.png')  <-----To download the graph

m
BBox=((data.Longitude.min(), data.Longitude.max(), data.Latitude.min(), data.Latitude.max()))

#defines the area of the map that will include all the spatial points 

BBox
#Creating a list with additional area to cover all points into the map

BBox=[-6.442, 2.100, 49.762, 54.858]

BBox
#import map layer extracted based on the lat and long values

data_map = plt.imread('../input/gb-map/map.png')



fig, ax = plt.subplots(figsize = (10,10))

ax.scatter(data.Longitude, data.Latitude, zorder=2, alpha= 0.8, c='lime', s=80, edgecolors='k')

ax.set_title('Protected Shipwrecks Sites in English territorial waters', fontsize=15)

ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])

ax.imshow(data_map, zorder=1, extent=BBox,aspect='auto')

plt.show()
# Total area of protected shipwreck sites

print("Total area of protected sites:",data.AREA_HA.sum())