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
import squarify    # pip install squarify (algorithm for treemap)
from mpl_toolkits.basemap import Basemap
import folium
import json

from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.sampledata.sample_geojson import geojson
df = pd.read_csv('../input/singapore-postal-code-to-latlon/SG_postal.csv')
df.head()
df.columns
x= df['street_name'].value_counts()
x.head(15)
plt.style.use('default')
sns.set_style("dark")

def plot_count(feature, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(6*size,8))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:10], palette='Set2', edgecolor='k')
    g.set_title("Number and percentage of postal codes by Area",fontsize=15)
    g.set_ylabel("Number of postal codes", fontsize=12)
    g.set_xlabel("Street Name", fontsize=13)
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show() 
plot_count("street_name", df, size=4)
# lat, long extreme(boundary) values
BBox=[103.5785, 104.1028, 1.1556, 1.4800]
#import map layer extracted based on the lat and long values
la_map = plt.imread('../input/singapore-map/map (1).png')

fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(df.lon,df.lat, zorder=2, alpha= 0.6, c='orange',edgecolors='r', s=30)
ax.set_title('Singapore Postal Codes', fontsize=20)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(la_map, zorder=1, extent=BBox,aspect='auto')
plt.show()
df_codes = gpd.read_file('../input/singapore-postal-code-to-latlon/SG_postal.csv')
df_codes
# Your code here: Create the GeoDataFrame
df = gpd.GeoDataFrame(df_codes, geometry=gpd.points_from_xy(df_codes["lon"].astype('float32'), df_codes["lat"].astype('float32')) )

# Your code here: Set the CRS to {'init': 'epsg:4326'}
df.crs = {'init': 'epsg:4326'}
df.head()
singapore = gpd.read_file('../input/singapore-shp/gadm36_SGP_1.shp')
singapore.head()
ax = singapore.plot(figsize=(20,20), color='lightgray', linestyle='-', edgecolors='k')
df.plot(ax=ax, markersize=5)

singapore1 = gpd.read_file('../input/singapore-geojson/national-map-polygon-geojson.geojson')
singapore1.head()
ax = singapore1.plot(figsize=(20,20), color='lightgray', linestyle='-', edgecolors='k')
df.plot(ax=ax, markersize=5, color='orange', alpha=0.6)