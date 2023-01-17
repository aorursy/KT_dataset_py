# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(os.listdir('../input'))
nRowsRead = 2500

df1 = pd.read_csv('../input/fukushima-daichii-seawater-radiation-data/1060106000_00.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = '/1060106000_00.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')




df1.info()



df1.columns
df1.describe()
df1.head(15)
plt.hist(df1['Sampling point ID'], color = 'blue', edgecolor = 'black',

         bins = int(180/5))
for i, binwidth in enumerate([10, 50, 75, 100]):

    

   

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Distance from TEPCO Fukushima Dai-ichi NPP (km)'], bins = int(200/binwidth),

             color = 'purple', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Distance from Fukushima Dai-ichi NPP (km)', size = 10)

    ax.set_ylabel('SeaWater', size=10)



plt.tight_layout()

plt.show()




plt.matshow(df1.corr())

plt.show()



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




df1['Sampling coordinate North latitude (Decimal)'].plot(kind = 'line', color = 'g',label = 'Latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df1['Sampling coordinate East longitude (Decimal)'].plot(color = 'r',label = 'Longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()



df1.plot(kind='scatter', x='Sampling coordinate North latitude (Decimal)', y='Sampling coordinate East longitude (Decimal)',alpha = 0.5,color = 'blue')

plt.xlabel('Latitude')              # label = name of label

plt.ylabel('Longitude')

plt.title('Latitude Longitude Scatter Plot')
for index,value in df1[['Distance from TEPCO Fukushima Dai-ichi NPP (km)']][0:100].iterrows():

    print(index," : ",value)
df1.boxplot(column='Sampling coordinate North latitude (Decimal)',by = 'Sampling coordinate East longitude (Decimal)',grid=True, rot=1000, fontsize=10,figsize=(25,15))
data1 = df1.loc[:,["Sampling coordinate North latitude (Decimal)","Sampling coordinate East longitude (Decimal)","Distance from TEPCO Fukushima Dai-ichi NPP (km)"]]

data1.plot()
data1.plot(subplots = True,figsize=(25,15))
data1.plot(kind = "scatter",x='Sampling coordinate North latitude (Decimal)', y='Sampling coordinate East longitude (Decimal)',figsize=(25,15))




data1.plot(kind = "hist",y = "Distance from TEPCO Fukushima Dai-ichi NPP (km)",bins = 50,range= (0,250))





import json

with open('../input/japan-geo/jp_prefs.geojson') as f:

    counties = json.load(f)

    

geo = json.dumps(json.load(open("../input/japan-geo/jp_prefs.geojson", 'r')))



data2 = [dict(type='scattergeo',lat = df1['Sampling coordinate North latitude (Decimal)'],lon = df1['Sampling coordinate East longitude (Decimal)'],

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Distance from TEPCO Fukushima Dai-ichi NPP (km)"], colorbar = dict(title='Distance from TEPCO Fukushima Dai-ichi NPP (km)')))]





layout1 = dict(title='fukushima-seawater-radiation',

              geo = dict(scope='asia',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))

import plotly

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

daf = pd.read_csv("../input/fukushima-daichii-seawater-radiation-data/1060106000_00.csv")

city=gpd.read_file("../input/countries/ne_10m_admin_0_countries.shp")

lat=daf['Sampling coordinate North latitude (Decimal)'].values

lon= daf['Sampling coordinate East longitude (Decimal)'].values



fig = plt.figure(figsize=(20, 20))

m = Basemap(projection='lcc', resolution='l', 

            lat_0=35.00 , lon_0=145 ,

            width=1.05E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='red',linewidth=3)

m.drawcountries(color='gray',linewidth=3)

m.drawstates(color='green')

m.scatter(lon,lat, latlon=True,s=10,

          cmap='YlGnBu_r', alpha=0.5)
latitude, longitude = 40, 10.0

map_seawater_Rad = folium.Map(location=[latitude, longitude], zoom_start=2)

lat=daf['Sampling coordinate North latitude (Decimal)'].values

lon= daf['Sampling coordinate East longitude (Decimal)'].values

dis=daf["Distance from TEPCO Fukushima Dai-ichi NPP (km)"]

viridis = cm.get_cmap('viridis', 1)



colors_array = viridis(np.arange(dis.min()-1,dis.max()))

rainbow=[ colors.rgb2hex(colors_array[dis,:]) for dis in range(colors_array.shape[0]) ]





for lat, lng in zip(lat, lon):

    label = 'Radiation Seawater Activity'

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=3,

        popup=label,

        color=rainbow[i-1],

        fill=True,

        fill_color=rainbow[i-1],

        fill_opacity=0.5).add_to(map_seawater_Rad)  



map_seawater_Rad