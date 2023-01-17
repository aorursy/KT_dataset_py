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

import squarify  

from mpl_toolkits.basemap import Basemap

import folium

import plotly.express as px

import plotly.graph_objects as go
#Great Ocean Road

gor=pd.read_csv('../input/tripadvisor-attractions-reviews-nearby-locations/attractionsnearbyLocations_GOR.csv')

gor.head()
# Sunshine Coast 

scc=pd.read_csv('../input/scc-cdsv/datasets_831344_1422351_attractionsnearbyLocations_SCC.csv')

scc.head()
qld = gpd.read_file('../input/aus-state-geojson-files/QLD.json')

qld.head()
vic = gpd.read_file('../input/aus-state-geojson-files/VIC.json')

vic.head()
# Create the GeoDataFrame

gor = gpd.GeoDataFrame(gor, geometry=gpd.points_from_xy(gor["attractionLongitude"], gor["attractionLatitude"]) )



# Set the CRS to {'init': 'epsg:4326'}

gor.crs = {'init': 'epsg:4326'}

gor.head()
# Using groupby to plot Attraction Sites accross the Great Ocean Road

ax=vic.plot(figsize=(15,10),linestyle="-",edgecolor='black',color='lightgray')

gor.plot(ax=ax, markersize=70, alpha=0.5, color='lime',edgecolor='k')

plt.title("Attraction Sites accross the Great Ocean Road (Victoria, Australia)", fontsize=12)

plt.show()
# Create the GeoDataFrame

scc = gpd.GeoDataFrame(scc, geometry=gpd.points_from_xy(scc["attractionLongitude"], scc["attractionLatitude"]) )



# Set the CRS to {'init': 'epsg:4326'}

scc.crs = {'init': 'epsg:4326'}

scc.head()
# Using groupby to plot Attraction Sites accross the Sunshine Coast

ax=qld.plot(figsize=(15,10),linestyle="-",edgecolor='black',color='lightgray')

scc.plot(ax=ax, markersize=70, alpha=0.5, color='orangered',edgecolor='k')

plt.title("Attraction Sites accross the Sunshine Coast (Queensland, Australia)", fontsize=12)

plt.show()
#Using Folium to plot attraction sites data on the map

# Make an empty map

m = folium.Map(location=[-38.253875, 143.476372], tiles="OpenStreetMap", zoom_start=9)

 

# I can add marker one by one on the map

for i in range(0,len(gor)):

    folium.Marker([gor.iloc[i]['attractionLatitude'], gor.iloc[i]['attractionLongitude']], popup=gor.iloc[i]['attractionName']).add_to(m)

#m.save('Folium GOR sites.png')  <-----To download the graph

m
#Using Folium to plot attraction sites data on the map

# Make an empty map

m = folium.Map(location=[-26.651284,153.066558], tiles="OpenStreetMap", zoom_start=10)

 

# I can add marker one by one on the map

for i in range(0,len(scc)):

    folium.Marker([scc.iloc[i]['attractionLatitude'], scc.iloc[i]['attractionLongitude']], popup=scc.iloc[i]['attractionName']).add_to(m)

#m.save('Folium SCC sites.png')  <-----To download the graph

m
#defines the area of the map that will include all the spatial points 

BBox1=((gor.attractionLongitude.min(), gor.attractionLongitude.max(), gor.attractionLatitude.min(), gor.attractionLatitude.max()))

BBox1
#Creating a list with additional area to cover all points into the map

BBox1=[142.0740, 144.8730, -38.9990, -37.8950]



BBox1
#import map layer extracted based on the lat and long values

gor_map = plt.imread('../input/maps-of-gor-and-scc/gor2.png')



fig, ax = plt.subplots(figsize = (24,9))

ax.scatter(gor.attractionLongitude, gor.attractionLatitude, zorder=2, alpha= 0.6, edgecolors='k', c='lime', s=70)

ax.set_title("Attraction Sites accross the Great Ocean Road (Victoria, Australia)", fontsize=16)

ax.set_xlim(BBox1[0],BBox1[1])

ax.set_ylim(BBox1[2],BBox1[3])

ax.imshow(gor_map, zorder=1, extent=BBox1, aspect='auto')

plt.show()
#defines the area of the map that will include all the spatial points 

BBox2=((scc.attractionLongitude.min(), scc.attractionLongitude.max(), scc.attractionLatitude.min(), scc.attractionLatitude.max()))

BBox2
#Creating a list with additional area to cover all points into the map

BBox2=[152.7350, 153.2450, -26.9997, -25.9000]



BBox2
#import map layer extracted based on the lat and long values

scc_map = plt.imread('../input/maps-of-gor-and-scc/scc1.png')



fig, ax = plt.subplots(figsize = (24,9))

ax.scatter(scc.attractionLongitude,scc.attractionLatitude, zorder=2, alpha= 0.6, edgecolors='k', c='orange', s=70)

ax.set_title("Attraction Sites accross the Sunshine Coast (Queensland, Australia)")

ax.set_xlim(BBox2[0],BBox2[1])

ax.set_ylim(BBox2[2],BBox2[3])

ax.imshow(scc_map, zorder=1, extent=BBox2, aspect='equal')

plt.show()