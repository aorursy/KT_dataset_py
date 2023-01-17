

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import folium

from folium import plugins

from folium.plugins import HeatMap

# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors



import json

with open('../input/ukranie/Ukraine.json') as f:

    counties = json.load(f)

    

geo = json.dumps(json.load(open("../input/ukranie/Ukraine.json", 'r')))

import plotly

import pandas as pd

from mpl_toolkits.basemap import Basemap

import geopandas as gpd
print(os.listdir('../input'))
nRowsRead = 2500

df1 = pd.read_csv('../input/chernobyl-accident-ivankov-background-radiation/6_Ivankov_background_radiation.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = '../input/chernobyl-accident-ivankov-background-radiation/6_Ivankov_background_radiation.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.info()
df1.columns



df1.describe()
df1.head(10)
plt.hist(df1['Code'], color = 'blue', edgecolor = 'black',

         bins = int(180/5))
plt.hist(df1['at_1m'], color = 'red', edgecolor = 'brown',

         bins = int(180/5))
plt.hist(df1['at_0.1m'], color = 'red', edgecolor = 'brown',

         bins = int(180/5))
plt.hist(df1['Analysis_type_1_or_2'], color = 'green', edgecolor = 'black',

         bins = int(180/5))



for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['at_0.1m']=pd.to_numeric(df1['at_0.1m'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['at_1m'], bins = int(250/binwidth),

             color = 'purple', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('at_1m', size = 10)

    ax.set_ylabel('Unit', size=10)



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
df1.latitude.plot(kind = 'line', color = 'g',label = 'latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df1.longitude.plot(color = 'r',label = 'longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()
df1.plot(kind='scatter', x='latitude', y='longitude',alpha = 0.5,color = 'blue')

plt.xlabel('Latitude')              # label = name of label

plt.ylabel('Longitude')

plt.title('Latitude Longitude Scatter Plot')
for index,value in df1[['Code']][0:100].iterrows():

    print(index," : ",value)
%matplotlib inline

df1.boxplot(column='latitude',by = 'longitude',grid=True, rot=25000, fontsize=10,figsize=(25,25))
data1 = df1.loc[:,["latitude","longitude","Analysis_type_1_or_2","Code","at_0.1m","at_1m"]]

data1.plot()
data1.plot(subplots = True,figsize=(25,15))
data1.plot(kind = "scatter",x="longitude",y = "latitude",figsize=(25,15))
data1.plot(kind = "hist",y = "Code",bins = 50,range= (0,250))
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Code",bins = 50,range= (0,250),ax = axes[0])

data1.plot(kind = "hist",y = "Code",bins = 50,range= (0,250),ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt














data2 = [dict(type='scattergeo',lat = df1.latitude,lon = df1.longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["at_1m"], colorbar = dict(title='at_1m')))]





layout1 = dict(title='chernobyl-at_1m',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))
%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})

data2 = [dict(type='scattergeo',lat = df1.latitude,lon = df1.longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["at_0.1m"], colorbar = dict(title='at_0.1m')))]





layout1 = dict(title='chernobyl-at_0.1m',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))
%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})
daf = pd.read_csv("../input/chernobyl-accident-ivankov-background-radiation/6_Ivankov_background_radiation.csv")

city=gpd.read_file("../input/countries/ne_10m_admin_0_countries.shp")

lat=daf['latitude'].values

lon= daf['longitude'].values



fig = plt.figure(figsize=(15, 15))

m = Basemap(projection='lcc', resolution='l', 

            lat_0=50 , lon_0=33 ,

            width=1.05E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='red',linewidth=3)

m.drawcountries(color='green',linewidth=3)

m.drawstates(color='brown')

m.scatter(lon,lat, latlon=True,s=10,

          cmap='YlGnBu_r', alpha=0.5)
latitude, longitude = 40, 10.0

map_che_Rad = folium.Map(location=[latitude, longitude], zoom_start=2)

lat=daf['latitude'].values

lon= daf['longitude'].values

dis=daf["Code"]



# set color scheme for the clusters

viridis = cm.get_cmap('viridis', dis.max())

colors_array = viridis(np.arange(dis.min()-1,dis.max()))

rainbow=[ colors.rgb2hex(colors_array[i,:]) for i in range(colors_array.shape[0]) ]

i=5

for lat, lng in zip(lat, lon):

    label = 'Radiation Background Activity'

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=3,

        popup=label,

        color=rainbow[i-1],

        fill=True,

        fill_color=rainbow[i-1],

        fill_opacity=0.5).add_to(map_che_Rad)  



map_che_Rad