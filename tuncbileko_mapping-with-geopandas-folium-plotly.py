# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium

import geopandas as gdp

import shapefile as shp

from shapely.geometry import Point, Polygon

import plotly.express as px

import json



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
child = pd.read_csv("../input/child-mortality/child_mortality_0_5_year_olds_dying_per_1000_born.csv")

world = gdp.read_file("../input/child-mortality/ne_10m_admin_0_countries.shp")

world_js=gdp.read_file("../input/child-mortality/ne_10geopack.geojson")

with open('../input/child-mortality/custom.geo.json') as world:

  data = json.load(world)



data["features"][0]
print(type(world_js))

world_js
world.reset_index()

world.head()
#Addding 2000 Child mortality rates to world coordinates dataset with left join. 

merged = pd.merge(world, 

                  child['2000'],

                  left_on='ADMIN',

                  right_on = child['country'],

                  how='left')





merged.head()

#Formatting merged data frame to geodata frame

gdf = gdp.GeoDataFrame(merged)
# create map

fig, ax = plt.subplots(figsize=(15,15))

gdf.plot(ax=ax)
# set a variable that will call whatever column we want to visualise on the map

variable = '2000'

# set the range for the choropleth

vmin, vmax = 0, 50

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 20))



# create map

bx=gdf.plot(column=variable, cmap='Oranges', linewidth=0.8,ax=ax, edgecolor='0.8', figsize=(20, 20))

ax.axis('off')

# add a title

ax.set_title('Child Mortality', fontdict={'fontsize': '25', 'fontweight' : '3'})



# Create colorbar as a legend

sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm,cax=fig.add_axes([0.85, 0.50, 0.009, 0.17]))
#Spatial reference system

gdf.crs = "EPSG:4326"
#creating map

map = folium.Map(location=[36.7538, 3.0588], zoom_start=2,min_zoom = 2, max_zoom =4,  tiles='stamenwatercolor')



#creating choropleth map with child mortality data

map.choropleth(

    geo_data=data,

    data=child,

    columns=['country', '2000'],

    key_on='feature.properties.name',

    fill_color='Oranges', 

    fill_opacity=1, 

    line_opacity=1,

    legend_name='Child Mortality Rate of 2000',

    smooth_factor=0)



map
fig = px.choropleth(child, locations="country", locationmode='country names',

                    color="2000", hover_name="country",

                    color_continuous_scale="Oranges",

                     labels={'Cases':'Cases'},

                     title='Child Mortality Rates 2000')

fig.show()

fig = px.choropleth_mapbox(child, geojson=data, locations='country', color='1998',

                           color_continuous_scale="Viridis",

                           mapbox_style="carto-positron",

                           featureidkey="properties.name",

                           zoom=1, center = {"lat": 20, "lon": 0},

                           opacity=0.5,

                           labels={'1998':'child mortalilty rate'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()