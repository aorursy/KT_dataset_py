# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




print(os.listdir('../input'))



nRowsRead = 3000

df1 = pd.read_csv('../input/chernobyl-exclusion-zone-soil-radiation/Soil_radionuclide_data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = '../input/chernobyl-exclusion-zone-soil-radiation/Soil_radionuclide_data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')




df1.info()



df1.columns
df1.describe()
df1.head(10)
plt.hist(df1['Cs-137_Soil_Bq_kg_DM'], color = 'blue', edgecolor = 'black',

         bins = int(180/5))
plt.hist(df1['Sr-90_Soil_Bq_kg_DM'], color = 'red', edgecolor = 'brown',

         bins = int(180/5))
plt.hist(df1['Am-241_Soil_Bq_kg_DM'], color = 'green', edgecolor = 'brown',

         bins = int(180/5))
plt.hist(df1['Pu-239_240_Soil_Bq_kg_DM'], color = 'purple', edgecolor = 'black',

         bins = int(180/5))
plt.hist(df1['Pu-238_Soil_Bq_kg_DM'], color = 'yellow', edgecolor = 'black',

         bins = int(180/5))
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Cs-137_Soil_Bq_kg_DM']=pd.to_numeric(df1['Cs-137_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Cs-137_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'blue', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Cs-137_Soil_Bq_kg_DM', size = 10)

    ax.set_ylabel('Unit', size=10)



plt.tight_layout()

plt.show()
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Am-241_Soil_Bq_kg_DM']=pd.to_numeric(df1['Am-241_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Am-241_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'green', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Am-241_Soil_Bq_kg_DM', size = 10)

    ax.set_ylabel('Unit', size=10)



plt.tight_layout()

plt.show()
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Sr-90_Soil_Bq_kg_DM']=pd.to_numeric(df1['Sr-90_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Sr-90_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'red', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Sr-90_Soil_Bq_kg_DM', size = 10)

    ax.set_ylabel('Unit', size=10)



plt.tight_layout()

plt.show()
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Pu-239_240_Soil_Bq_kg_DM']=pd.to_numeric(df1['Pu-239_240_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Pu-239_240_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'brown', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Sr-90_Soil_Bq_kg_DM', size = 10)

    ax.set_ylabel('Unit', size=10)



plt.tight_layout()

plt.show()
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Pu-239_240_Soil_Bq_kg_DM']=pd.to_numeric(df1['Pu-239_240_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Pu-239_240_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'purple', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Sr-90_Soil_Bq_kg_DM', size = 10)

    ax.set_ylabel('Unit', size=10)



plt.tight_layout()

plt.show()
for i, binwidth in enumerate([10, 50, 75, 100]):

    

    df1['Pu-238_Soil_Bq_kg_DM']=pd.to_numeric(df1['Pu-238_Soil_Bq_kg_DM'])

    ax = plt.subplot(2, 2, i + 1)

    

    ax.hist(df1['Pu-238_Soil_Bq_kg_DM'], bins = int(250/binwidth),

             color = 'yellow', edgecolor = 'black')

    

    # Title and labels

    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)

    ax.set_xlabel('Sr-90_Soil_Bq_kg_DM', size = 10)

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




df1.Latitude.plot(kind = 'line', color = 'g',label = 'latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df1.Longitude.plot(color = 'r',label = 'longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()







df1.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.5,color = 'blue')

plt.xlabel('Latitude')              # label = name of label

plt.ylabel('Longitude')

plt.title('Latitude Longitude Scatter Plot')



for index,value in df1[['Dry_mass_of_soil_in_grams']][0:100].iterrows():

    print(index," : ",value)




%matplotlib inline

df1.boxplot(column='Latitude',by = 'Longitude',grid=True, rot=25000, fontsize=10,figsize=(25,25))



data1 = df1.loc[:,["N","Latitude","Longitude","Dry_mass_of_soil_in_grams","Cs-137_Soil_Bq_kg_DM","Sr-90_Soil_Bq_kg_DM","Am-241_Soil_Bq_kg_DM","Pu-239_240_Soil_Bq_kg_DM","Pu-238_Soil_Bq_kg_DM"]]

data1.plot()
data1.plot(subplots = True,figsize=(25,15))




data1.plot(kind = "scatter",x="Longitude",y = "Latitude",figsize=(25,15))







data2 = [dict(type='scattergeo',lat = df1.Latitude,lon = df1.Longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Am-241_Soil_Bq_kg_DM"], colorbar = dict(title='Am-241_Soil_Bq_kg_DM')))]





layout1 = dict(title='chernobyl-Am-241_Soil_Bq_kg_DM',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))







%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})



data2 = [dict(type='scattergeo',lat = df1.Latitude,lon = df1.Longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Sr-90_Soil_Bq_kg_DM"], colorbar = dict(title='Sr-90_Soil_Bq_kg_DM')))]





layout1 = dict(title='chernobyl-Sr-90_Soil_Bq_kg_DM',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(248,248,248)",subunitcolor = "rgb(215,215,215)",

                     countrycolor = "rgb(87,87,87)",countrywidth =0.8, subunitwidth=0.5))


%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})
data2 = [dict(type='scattergeo',lat = df1.Latitude,lon = df1.Longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Pu-239_240_Soil_Bq_kg_DM"], colorbar = dict(title='Pu-239_240_Soil_Bq_kg_DM')))]





layout1 = dict(title='chernobyl-Pu-239_240_Soil_Bq_kg_DM',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))
%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})
data2 = [dict(type='scattergeo',lat = df1.Latitude,lon = df1.Longitude,

             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',

            color = df1["Pu-238_Soil_Bq_kg_DM"], colorbar = dict(title='Pu-238_Soil_Bq_kg_DM')))]





layout1 = dict(title='chernobyl-Pu-238_Soil_Bq_kg_DM',

              geo = dict(scope='europe',showland = True,

                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",

                     countrycolor = "rgb(85,85,85)",countrywidth =0.8, subunitwidth=0.5))
%matplotlib notebook

init_notebook_mode(connected = True)

fig = go.Figure(data=data2, layout=layout1)

plotly.offline.iplot({"data": data2,"layout": layout1})
daf = pd.read_csv("../input/chernobyl-exclusion-zone-soil-radiation/Soil_radionuclide_data.csv")

city=gpd.read_file("../input/countries/ne_10m_admin_0_countries.shp")

lat=daf['Latitude'].values

lon= daf['Longitude'].values



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

daf.dropna(subset=['Latitude'], how='all', inplace=True)

daf.dropna(subset=['Longitude'], how='all', inplace=True)

lat=daf['Latitude'].values

lon= daf['Longitude'].values

# set color scheme for the clusters

viridis = cm.get_cmap('viridis', 5)

colors_array = viridis(np.arange(1,5))

rainbow=[ colors.rgb2hex(colors_array[i,:]) for i in range(colors_array.shape[0]) ]

i=1

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

        fill_opacity=0.5).add_to(map_che_Rad)  



map_che_Rad