# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gdp

import seaborn as sns

import matplotlib 

import shapefile as shp

from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt

from pyproj import Proj, transform

import pyproj

import geopandas 

import mplleaflet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pop_2017 = geopandas.read_file("../input/population-2017/istanbul_mahalle_bazli_2017_nufus.dbf")

shp_tweet= geopandas.read_file("../input/shapefile-deneme/norm2_18apl17.shp")

Pandas_Table=geopandas.read_file("../input/normtwitter/norm2_18apl17.dbf")

inProj = pyproj.Proj(init='epsg:4326')

# resulting projection, WGS84, long, lat

outProj = pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=30 +k=1 +x_0=500000 +y_0=0 +a=6378137.0 +b=6356752.3142  +units=m +no_defs")

y = []



for xy in zip( Pandas_Table["lon"], Pandas_Table["lat"]):

    a = transform(inProj,outProj,xy[0],xy[1])

    y.append(Point(a))

    

geometry = y

crs = {'init': 'epsg:4326'}

Pandas_Table = gdp.GeoDataFrame(Pandas_Table,

                         crs=crs,

                         geometry=geometry)
pop_2017.head()
Pandas_Table.head()
liste=[0,0]

liste[0]=pop_2017['nufus17'].max()

liste[1]=pop_2017['nufus17'].min()

liste
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar



# set a variable that will call whatever column we want to visualise on the map

variable = 'nufus17'

# set the range for the choropleth

vmin, vmax = 0, 100000

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 20))

# create map

pop_2017.plot(column=variable, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

# add a title

ax.set_title('İstanbul 2017 Yılı Nufus Verisi', fontdict={'fontsize': '25', 'fontweight' : '3'})



# Create colorbar as a legend

sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm,cax=fig.add_axes([0.85, 0.50, 0.009, 0.17]))
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size





# set a variable that will call whatever column we want to visualise on the map

variable = 'nufus17'

# set the range for the choropleth

vmin, vmax = 0, 100000

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 20))



# create map

bx=pop_2017.plot(column=variable, cmap='Oranges', linewidth=0.8,ax=ax, edgecolor='0.8', figsize=(20, 20))

Pandas_Table.plot(ax=bx, alpha=0.1,color='cornflowerblue',markersize=1)

ax.axis('off')

# add a title

ax.set_title('İstanbul 2017 Yılı Nufus Verisi', fontdict={'fontsize': '25', 'fontweight' : '3'})



# Create colorbar as a legend

sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm,cax=fig.add_axes([0.85, 0.50, 0.009, 0.17]))
ax = pop_2017.plot(figsize=(20, 20),alpha='0.4')

Pandas_Table.plot(ax=ax, alpha=0.3,color='g',markersize=1)

ax.set_title('İstanbul Nisan 2017 Tweet Dağılımı', fontdict={'fontsize': '25', 'fontweight' : '3'})
pop_2017['geoid'] = pop_2017.index.astype(str)

data= pop_2017[['geoid','nufus17','geometry']]



data.head()

data = data.loc[(data['nufus17'] > 0) & (data['nufus17'] <= 500)]

data = data.to_crs(epsg=4326)

print(data.crs)

data.info()

import folium



m = folium.Map(location=[41.109550, 28.989620], tiles = 'cartodbpositron', zoom_start=10, control_scale=True)



folium.Choropleth(

    geo_data=data,

    name='Population in 2017',

    data=data,

    columns=['geoid', 'nufus17'],

    key_on='feature.id',

    fill_color='YlOrRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    line_color='white',

    line_weight=0,

    highlight=False,

    smooth_factor=1.0,

    #threshold_scale=[100, 250, 500, 1000, 2000],

    legend_name= 'Population in İstanbul').add_to(m)



m
shp_tweet.head()
shp_tweet['lon'] = shp_tweet["geometry"].x

shp_tweet['lat'] = shp_tweet["geometry"].y

shp_tweet = shp_tweet.loc[(shp_tweet['lon'] > 29.024876) & (shp_tweet['lon'] <= 29.084876)]



points_array = shp_tweet[['lat', 'lon']].values





print(type(points_array))

print(points_array[:5])

from folium.plugins import HeatMap

h = folium.Map(location=[41.109550, 28.989620], tiles = 'cartodbpositron', zoom_start=10, control_scale=True)

HeatMap(points_array).add_to(h)



#showheatmap

h
data.plot()

#mplleaflet.show()
ax = data.plot(markersize = 10, color = "red")

mplleaflet.display(fig=ax.figure, crs=data.crs)