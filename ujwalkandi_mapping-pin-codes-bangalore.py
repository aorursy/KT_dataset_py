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
import matplotlib
import seaborn as sns
import geopandas as gpd
import squarify   
from mpl_toolkits.basemap import Basemap
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
df = pd.read_excel('../input/bangaloreareaspincodewithlatitudelongitude/BangaloreAreaLatLongDetails.xlsx')
df.head()
df.Area.value_counts().value_counts()
#defines the area of the map that will include all the spatial points 
BBox=((df.Longitude.min(), df.Longitude.max(), df.Latitude.min(), df.Latitude.max()))
BBox
#Creating a list with additional area to cover all points into the map
BBox=[ 77.4123, 77.7831, 12.8131, 13.1885]
BBox
#import map layer extracted based on the lat and long values
b_map = plt.imread('../input/bengaluru-map/map (1).png')

fig, ax = plt.subplots(figsize = (10,8))
ax.scatter(df.Longitude,df.Latitude, zorder=2, alpha= 0.8, c='orange',edgecolors='k', s=90)
ax.set_title('Bengaluru Pin Codes', fontsize=20)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(b_map, zorder=1, extent=BBox,aspect='auto')
plt.show()
#Using Folium to plot World Heritage Sites data on the map
# Make an empty map
m = folium.Map(location=[12.983668, 77.588324], tiles="OpenStreetMap", zoom_start=12)
 
# I can add marker one by one on the map
for i in range(0,len(df)):
    folium.Marker([df.iloc[i]['Latitude'], df.iloc[i]['Longitude']], popup=df.iloc[i]['Area']).add_to(m)
#m.save('Folium BBMP.png')  <-----To download the graph
m
m_5 = folium.Map(location=[12.983668, 77.588324], tiles='cartodbpositron', zoom_start=12)

# Add a heatmap to the base map
HeatMap(data=df[['Latitude', 'Longitude']], radius=30).add_to(m_5)

# Display the map
m_5
# Create the GeoDataFrame
df2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]) )

# Set the CRS to {'init': 'epsg:4326'}
df2.crs = {'init': 'epsg:4326'}
df2.head()
df1=gpd.read_file('../input/bangalore-map/BBMP.GeoJSON.txt')
df1.head()
# Plot points over world layer
ax = df1.plot(figsize=(15,10), linestyle=':', edgecolor='black', color='lightgray')
df2.plot(ax=ax, markersize=80, alpha=0.8,edgecolors='k', c='orange')
plt.title("Pin Codes that come under the BBMP(Bengaluru Municipal Corporation)", fontsize=13)
plt.show()