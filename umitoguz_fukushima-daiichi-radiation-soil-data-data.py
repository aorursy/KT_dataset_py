# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from folium import plugins

from folium.plugins import HeatMap

# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting



import os # accessing directory structure



# Any results you write to the current directory are saved as output.
import os

print(os.listdir('../input'))
import pandas as pd

nRowsRead = 2500

df1 = pd.read_csv('../input/fukushima-daiichi-soil-radiation-data/FieldSampleSoilResults_2.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = '/FieldSampleSoilResults_2.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.info()
df1.columns
df1.describe()
df1.head(10)
import matplotlib.pyplot as plt 

plt.hist(df1['Analysis Id'], color = 'blue', edgecolor = 'black',

         bins = int(180/5))

plt.hist(df1['Sample Id'], color = 'red', edgecolor = 'brown',

         bins = int(180/5))

plt.hist(df1['Bearing'], color = 'green', edgecolor = 'black',

         bins = int(180/5))

for i, binwidth in enumerate([10, 50, 75, 100]):

    

   

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Distance(miles)'], bins = int(200/binwidth),

             color = 'purple', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Distance(miles)', size = 10)

    ax.set_ylabel('Soil', size=10)



plt.tight_layout()

plt.show()
plt.matshow(df1.corr())

plt.show()
import seaborn as sns

corr = df1.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=50,

    horizontalalignment='right'

);
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df1.Latitude.plot(kind = 'line', color = 'g',label = 'Latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df1.Longitude.plot(color = 'r',label = 'Longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()
df1.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.5,color = 'blue')

plt.xlabel('Result')              # label = name of label

plt.ylabel('Longitude')

plt.title('Result Longitude Scatter Plot')
for index,value in df1[['Distance(miles)']][0:100].iterrows():

    print(index," : ",value)
df1.boxplot(column='Latitude',by = 'Longitude',grid=True, rot=1000, fontsize=10,figsize=(25,15))
data1 = df1.loc[:,["Latitude","Longitude","Distance(miles)","Result"]]

data1.plot()
data1.plot(subplots = True,figsize=(25,15))
data1.plot(kind = "scatter",x="Longitude",y = "Latitude",figsize=(25,15))
data1.plot(kind = "hist",y = "Distance(miles)",bins = 50,range= (0,250),normed = True)


fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Result",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Result",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
import folium

import json

with open('../input/japan-geo/jp_prefs.geojson') as f:

    counties = json.load(f)

    

geo = json.dumps(json.load(open("../input/japan-geo/jp_prefs.geojson", 'r')))

import plotly

import pandas as pd













data2 = [dict(type='scattergeo',lat = df1.Latitude,lon = df1.Longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Distance(miles)"], colorbar = dict(title='Distance(miles)')))]





layout1 = dict(title='fukushima-soil-radiation',

              geo = dict(scope='asia',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import geopandas as gpd

import pandas as pd

daf = pd.read_csv("../input/fukushima-daiichi-soil-radiation-data/FieldSampleSoilResults_2.csv")

city=gpd.read_file("../input/countries/ne_10m_admin_0_countries.shp")

lat=daf['Latitude'].values

lon= daf['Longitude'].values



fig = plt.figure(figsize=(15, 15))

m = Basemap(projection='lcc', resolution='l', 

            lat_0=35.00 , lon_0=136 ,

            width=1.05E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='red',linewidth=3)

m.drawcountries(color='gray',linewidth=3)

m.drawstates(color='green')

m.scatter(lon,lat, latlon=True,s=10,

          cmap='YlGnBu_r', alpha=0.5)
latitude, longitude = 40, 10.0

map_soil_Rad = folium.Map(location=[latitude, longitude], zoom_start=2)

lat=daf['Latitude'].values

lon= daf['Longitude'].values

dis=daf["Distance(miles)"]

a=dis//2.1

# set color scheme for the clusters

viridis = cm.get_cmap('viridis', dis.max())

colors_array = viridis(np.arange(dis.min()-1,dis.max()))

rainbow=[ colors.rgb2hex(colors_array[i,:]) for i in range(colors_array.shape[0]) ]

for lat, lng in zip(lat, lon):

    label = 'Radiation Soil Activity'

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=3,

        popup=label,

        color=rainbow[i-1],

        fill=True,

        fill_color=rainbow[i-1],

        fill_opacity=0.5).add_to(map_soil_Rad)  



map_soil_Rad