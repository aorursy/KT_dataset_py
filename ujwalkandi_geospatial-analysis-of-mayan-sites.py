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
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import squarify    
import math
from mpl_toolkits.basemap import Basemap
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
df = pd.read_csv("../input/mayan-site/scrapedData.csv", engine='python')
df.head()
df.columns
df['Country_Area'].value_counts()[:12]
plt.figure(figsize=(14,7))
sns.set(style="white")
colors = ['red','orange','orange','orange','orange','green','green','orange','green','red','blue']
df['Country_Area'].value_counts()[:11].plot(kind='bar',edgecolor='k', color=colors, alpha=0.8)
  
for index, value in enumerate(df['Country_Area'].value_counts()[:11]):
    plt.text(index, value, str(value))
plt.xlabel("Countries, Region/Province", fontsize=14)
plt.ylabel("Count of Sites", fontsize=13)
plt.title("Maya Sites by Country", fontsize=18)
plt.show()
df.Longitude[62]=-89.486376
df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].apply(pd.to_numeric) 
#defines the area of the map that will include all the spatial points 
BBox=((df.Latitude.min(), df.Latitude.max(), df.Longitude.min(), df.Longitude.max()))
BBox
#Creating a list with additional area to cover all points into the map
BBox=[ -94.768, -85.781, 13.710, 22.313]

BBox
#import map layer extracted based on the lat and long values
la_map = plt.imread('../input/map-maya/map (1).png')

fig, ax = plt.subplots(figsize = (24,9))
ax.scatter(df.Longitude,df.Latitude, zorder=2, alpha= 0.6, edgecolors='k', c='orange', s=70)
ax.set_title('Maya Sites across Central America')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(la_map, zorder=1, extent=BBox, aspect='equal')
plt.show()
# to remove Nan from the columns
df.dropna(inplace=True)
m_5 = folium.Map(location=[17.462089, -90.443718], tiles='cartodbpositron', zoom_start=6.5)

# Add a heatmap to the base map
HeatMap(data=df[['Latitude', 'Longitude']], radius=30).add_to(m_5)

# Display the map
m_5
#  Create the GeoDataFrame
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]) )

# Set the CRS to {'init': 'epsg:4326'}
df.crs = {'init': 'epsg:4326'}
df.head()
belize = gpd.read_file('../input/central-americas/gadm36_BLZ_1.shp')
belize.crs = {'init': 'epsg:4326'}
belize.head()
guatemala = gpd.read_file('../input/central-americas/gadm36_GTM_2.shp')
guatemala.crs = {'init': 'epsg:4326'}
guatemala.head()
honduras = gpd.read_file('../input/central-americas/gadm36_HND_1.shp')
honduras.crs = {'init': 'epsg:4326'}
honduras.head()
mexico = gpd.read_file('../input/central-americas/gadm36_MEX_2.shp')
mexico.crs = {'init': 'epsg:4326'}
mexico.head()
joined = belize.geometry.append(guatemala.geometry)
joined
j1=joined.geometry.append(honduras.geometry)
j1
j2=j1.geometry.append(mexico.geometry)
j2
ax = j2.plot(figsize=(20,20), color='lightgray', linestyle='-', edgecolors='k')
df.plot(ax=ax, markersize=50, color='yellow', edgecolors='k')