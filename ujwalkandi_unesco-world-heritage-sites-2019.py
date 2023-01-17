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


print(gpd.__version__)
whc = pd.read_csv('../input/unesco-world-heritage-sites/whc-sites-2019.csv')
whc.head()
#Selecting the first 20 countries with the most World Heritage Sites
whc['states_name_en'].value_counts()[:20]
plt.figure(figsize=(20,9))
whc['states_name_en'].value_counts()[:20].plot(kind='bar',edgecolor='k', alpha=0.8)
  
for index, value in enumerate(whc['states_name_en'].value_counts()[:20]):
    plt.text(index, value, str(value))
plt.xlabel("Countries", fontsize=14)
plt.ylabel("Count of sites", fontsize=13)
plt.title("World Heritage Sites by Country", fontsize=15)
plt.show()
# Considering countries with World Heritage Sites that have a area of more than 7200000 hectares.
whc_range = whc[whc["area_hectares"] >= 7200000]

#Utilise matplotlib to scale our goal numbers between the min and max, then assign this scale to our values.
norm = matplotlib.colors.Normalize(vmin=min(whc_range.area_hectares), vmax=max(whc_range.area_hectares))
colors = [plt.cm.YlGnBu(norm(value)) for value in whc_range.area_hectares]

#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(18,8)

#Use squarify to plot our data, label it and add colours. We add an alpha layer to ensure black labels show through
squarify.plot(label=whc_range.states_name_en.unique(),sizes=whc_range.area_hectares, color = colors, alpha=.9,edgecolor='k')
plt.title("World Heritage Sites by Area (hectares)",fontsize=16)

#Remove our axes and display the plot
plt.axis('off')
plt.show()

#Using Basemap(matplotlib) to plot spatial data on the Map
plt.figure(figsize=(32,9))
m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.9, lake_color='white')
m.drawcoastlines(linewidth=0.5, color="k")

# Add a marker per city of the data frame!
m.plot(whc['longitude'], whc['latitude'], linestyle='none', marker="o", markersize=12, alpha=1, c="orange", markeredgecolor="black", markeredgewidth=0.9)


# copyright and source data info
plt.text( -155, -58,'UNESCO World Heritage Sites 2019\n\nData collected from whc.unesco.org\nPlot realized with Python and the Basemap library.', ha='left', va='bottom', size=8, color='#555555' )
plt.title("UNESCO World Heritage Sites 2019", fontsize=13)
plt.show()
plt.figure(figsize=(32,9))
m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)
m.bluemarble()

# Add a marker per city of the data frame!
m.plot(whc['longitude'], whc['latitude'], linestyle='none', marker="o", markersize=10, alpha=1, c="orange", markeredgecolor="black", markeredgewidth=0.9)
plt.title("UNESCO World Heritage Sites 2019", fontsize=13)
plt.show()
# Set the dimension of the figure
my_dpi=96
plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)
 
# Make the background map
m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="k")
 
# prepare a color for each point depending on the continent.
whc['states_name_en'] = pd.factorize(whc['region_en'])[0]
 
# Add a point per position
m.scatter(whc['longitude'], whc['latitude'], alpha=0.8, c=whc['states_name_en'], cmap="Set1", edgecolors='k',marker="o")
 
# copyright and source data info
plt.text( -170, -58,'UNESCO World Heritage Sites 2019\n\nData collected from whc.unesco.org\nPlot realized with Python and the Basemap library.', ha='left', va='bottom', size=9, color='#555555' )
 
# Save as png
# plt.savefig('#unesco_world_heritage_sites.png', bbox_inches='tight')
plt.title("World Heritage Sites by Region", fontsize=18)
plt.show()
# Set the dimension of the figure
my_dpi=96
plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)
 
# Make the background map
m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="k")
 
# prepare a color for each point depending on the continent.
x = whc['states_name_en'].copy()
x = pd.factorize(whc['category'])[0]
 
# Add a point per position
m.scatter(whc['longitude'], whc['latitude'], alpha=0.8, c=x, cmap="Set1", edgecolors='k',marker="o")
 
# copyright and source data info
plt.text( -170, -58,'UNESCO World Heritage Sites 2019\n\nData collected from whc.unesco.org\nPlot realized with Python and the Basemap library.', ha='left', va='bottom', size=9, color='#555555' )
 
# Save as png
# plt.savefig('#unesco_world_heritage_sites.png', bbox_inches='tight')
plt.title("World Heritage Sites by Category", fontsize=18)
plt.show()
#Using Folium to plot World Heritage Sites data on the map
# Make an empty map
m = folium.Map(location=[20, 0], tiles="OpenStreetMap", zoom_start=2.4)
 
# I can add marker one by one on the map
for i in range(0,len(whc)):
    folium.Marker([whc.iloc[i]['latitude'], whc.iloc[i]['longitude']], popup=whc.iloc[i]['name_en']).add_to(m)
#m.save('Folium UNESCO sites 2019.png')  <-----To download the graph
m
# BBox=((whc.latitude.min(), whc.latitude.max(), whc.longitude.min(), whc.longitude.max()))
#defines the area of the map that will include all the spatial points 

#Creating a list with additional area to cover all points into the map
BBox=[-180.0, 180.0, -59.0, 77.0]

BBox
#import map layer extracted based on the lat and long values
whc_map = plt.imread('../input/world-map/map (1).png')

fig, ax = plt.subplots(figsize = (24,13))
ax.scatter(whc.longitude, whc.latitude, zorder=2, alpha= 0.7, c='orange', s=60, edgecolors='k')
ax.set_title('UNESCO World Heritage Sites 2019', fontsize=20)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(whc_map, zorder=1, extent=BBox,aspect='auto')
plt.show()
# Create the GeoDataFrame
whs = gpd.GeoDataFrame(whc, geometry=gpd.points_from_xy(whc["longitude"], whc["latitude"]) )

# Set the CRS to {'init': 'epsg:4326'}
whs.crs = {'init': 'epsg:4326'}
# GeoDataFrame with an additional column (geometry)
whs.head()
# Load a GeoDataFrame with country boundaries 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Plot points over world layer
ax = world.plot(figsize=(15,10), linestyle=':', edgecolor='black', color='lightgray')
whs.plot(ax=ax, markersize=25, alpha=0.5)
plt.title("World Heritage Sites 2019", fontsize=12)
plt.show()
# Using groupby to plot World Heritage Sites by Category
ax=world.plot(figsize=(15,10),linestyle=":",edgecolor='black',color='lightgray')
whs.groupby("category")['geometry'].plot(ax=ax,marker='o', markersize=50, alpha=0.5)
plt.title("World Heritage Sites by Category", fontsize=12)
plt.show()
#Sites that are in danger - 1 
whc['danger'].value_counts()
# Using groupby to plot World Heritage Sites that are in the danger list
ax=world.plot(figsize=(15,10),linestyle=":",edgecolor='black',color='lightgray')
whs.groupby("danger")['geometry'].plot(ax=ax,marker='o', markersize=50, alpha=0.5)
plt.title("World Heritage Sites that are listed in the Danger list", fontsize=12)
plt.show()