# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium # map rendering library

from folium import plugins

from folium.plugins import HeatMap

# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir('../input'))


nRowsRead = 2000

df1 = pd.read_csv('../input/fukushima-river-radiation/1030106000_00.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = '/fukushima-river-radiation/1030106000_00.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.info()
df1.columns
df1.describe()
df1.head(10)
import matplotlib.pyplot as plt 

plt.hist(df1['Distance from TEPCO Fukushima Dai-ichi NPP (km)'], color = 'blue', edgecolor = 'black')

plt.hist(df1['Distance from TEPCO Fukushima Dai-ichi NPP (km)'], color = 'red', edgecolor = 'brown')
for i, binwidth in enumerate([10, 50, 75, 100]):

    

   

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Distance from TEPCO Fukushima Dai-ichi NPP (km)'], bins = int(200/binwidth),

             color = 'purple', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 9)

    ax.set_xlabel(' (km)', size =9)

    ax.set_ylabel('Binwidth', size=9)



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
df1['Distance from TEPCO Fukushima Dai-ichi NPP (km)'].plot(kind = 'line', color = 'g',label = 'Distance from TEPCO Fukushima Dai-ichi NPP (km)',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df1['137Cs Detection limit (Bq/kg)'].plot(color = 'r',label = '137Cs Detection limit (Bq/kg)',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()



df1.plot(kind='scatter', x='137Cs Detection limit (Bq/kg)', y='134Cs Detection limit (Bq/kg)',alpha = 0.5,color = 'blue')

plt.xlabel('137Cs Detection limit (Bq/kg)')              # label = name of label

plt.ylabel('134Cs Detection limit (Bq/kg)')

plt.title('137Cs Detection limit (Bq/kg) 134Cs Detection limit (Bq/kg) Scatter Plot')
for index,value in df1[['Sampling coordinate North latitude']][0:100].iterrows():

    print(index," : ",value)

    

for index,value in df1[['Sampling coordinate East longitude']][0:100].iterrows():

    print(index," : ",value)
df1.boxplot(column='Sampling coordinate North latitude (Decimal)',by = 'Sampling coordinate East longitude (Decimal)',grid=True, rot=1000, fontsize=10,figsize=(25,15))
data1 = df1.loc[:,["Sampling coordinate North latitude","Sampling coordinate East longitude","Sampling coordinate North latitude (Decimal)","Sampling coordinate East longitude (Decimal)"]]

data1.plot()
data1.plot(subplots = True,figsize=(25,15))
data1.plot(kind = "scatter",x="Sampling coordinate North latitude (Decimal)",y = "Sampling coordinate East longitude (Decimal)",figsize=(25,15))
data1.plot(kind = "hist",y = "Sampling coordinate North latitude (Decimal)",bins = 50,range= (0,250),normed = True)




fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Sampling coordinate North latitude (Decimal)",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Sampling coordinate East longitude (Decimal)",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt



import folium

import json

with open('../input/japan-geo/jp_prefs.geojson') as f:

    counties = json.load(f)

    

geo_str = json.dumps(json.load(open("../input/japan-geo/jp_prefs.geojson", 'r')))

import plotly

import pandas as pd







df = pd.read_csv("../input/fukushima-river-radiation/1030106000_00.csv")





data1 = [dict(type='scattergeo',lat = df["Sampling coordinate North latitude (Decimal)"],lon = df["Sampling coordinate East longitude (Decimal)"],

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df["Distance from TEPCO Fukushima Dai-ichi NPP (km)"], colorbar = dict(title='Distance from TEPCO Fukushima Dai-ichi NPP (km)')))]





layout1 = dict(title='fukushima-river-radiation',

              geo = dict(scope='asia',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.7, subunitwidth=0.5))
plotly.offline.iplot({

    "data": data1,

    "layout": layout1

})


import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import geopandas as gpd

import pandas as pd

daf = pd.read_csv("../input/fukushima-river-radiation/1030106000_00.csv")

city=gpd.read_file("../input/countries/ne_10m_admin_0_countries.shp")

lat=daf['Sampling coordinate North latitude (Decimal)'].values

lon= daf['Sampling coordinate East longitude (Decimal)'].values



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

map_water_Rad = folium.Map(location=[latitude, longitude], zoom_start=2)

lat=daf['Sampling coordinate North latitude (Decimal)'].values

lon= daf['Sampling coordinate East longitude (Decimal)'].values

# set color scheme for the clusters

viridis = cm.get_cmap('viridis', lat.max())

colors_array = viridis(np.arange(lat.min()-1,lat.max()))

rainbow=[ colors.rgb2hex(colors_array[i,:]) for i in range(colors_array.shape[0]) ]

i=2

for lat, lng in zip(lat, lon):

    label = 'Radiation River Activity'

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=3,

        popup=label,

        color=rainbow[i-1],

        fill=True,

        fill_color=rainbow[i-1],

        fill_opacity=0.5).add_to(map_water_Rad)  



map_water_Rad