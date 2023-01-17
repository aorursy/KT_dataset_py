# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import geopandas as gpd

import folium

from folium import Marker, Circle, PolyLine

from folium.plugins import HeatMap, MarkerCluster

import math
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train.head(10)
train.isnull().sum()
train=train.rename(columns={'Province/State': 'Province','Country/Region': 'Country',

                           'ConfirmedCases': 'Confirmed'})

train.head(3)
test.head(10)
test=test.rename(columns={'Province/State': 'Province','Country/Region': 'Country'})
test.head(10)
#train data confirmed case mean and sum in world 

print("confirmed mean:", train.Confirmed.mean())

print("confirmed sum:", train.Confirmed.sum())

print("Fatalities mean:", train.Fatalities.mean())

print("Fatalities sum:", train.Fatalities.sum())
datastata = train.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()

datastata

fig = go.Figure()

fig.add_trace(go.Scatter(

                x=datastata.Date,

                y=datastata['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=datastata.Date,

                y=datastata['Fatalities'],

                name="Fatalities",

                line_color='dimgray',

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(xaxis_range=['2020-01-22','2020-03-21'],

                  title_text="Corona time visualization")

fig.show()
Country=train.groupby(["Country"], as_index=False)["Confirmed","Fatalities"].sum().sort_values(by=["Confirmed","Fatalities"] ,ascending=False)

Country.head(10)
fig = px.bar(Country.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Country',

             title='World Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()
fig = px.bar(Country.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Country',

             title='World Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
#South korea  case  

cor = train.loc[train.Country == "Korea, South"]
core = cor.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()

fig = px.bar(core.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Date',

             title='South Korea Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(core.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Date',

             title='South Korea Confirmed case', text='Fatalities', height=1000, orientation='h')

fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus'])
#eu = ['Italy', 'Spain', 'France' , 'Netherlands', 'Germany', 'Switzerland', 'Beligum', 'Sweden' , 'Norway', 'Denmark']

eua = train.loc[train.Country.isin(europe)]

euc = eua.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()

euc.head(10)
fig = px.bar(core.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Date',

             title='EU Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(core.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Date',

             title='EU Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
euu=eua.groupby(["Country"], as_index=False)["Confirmed","Fatalities"].sum().sort_values(by=["Confirmed","Fatalities"] ,ascending=False)

fig = px.bar(euu.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Country',

             title='Every EU Country Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(euu.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Country',

             title='Every EU Country Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
trump = ['US','Canada']

america = train.loc[train.Country.isin(trump)]

usa = america.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()

usa.head(3)
fig = px.bar(usa.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Date',

             title='EU Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(usa.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Date',

             title='EU Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
us=america.groupby(["Country"], as_index=False)["Confirmed","Fatalities"].sum().sort_values(by=["Confirmed","Fatalities"] ,ascending=False)

fig = px.bar(us.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Country',

             title='EU Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(us.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Country',

             title='EU Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
mec = ['Iran','Iraq']

mec = train.loc[train.Country.isin(mec)]

meci = mec.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()

meci.head(3)
fig = px.bar(meci.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Date',

             title='Middle East Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(meci.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Date',

             title='Middle East Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
mecic=mec.groupby(["Country"], as_index=False)["Confirmed","Fatalities"].sum().sort_values(by=["Confirmed","Fatalities"] ,ascending=False)

fig = px.bar(mecic.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Country',

             title='Middle East Confirmed case', text='Confirmed', height=1000, orientation='h')

fig.show()



fig = px.bar(mecic.sort_values('Fatalities', ascending=False)[:20][::-1], 

             x='Fatalities', y='Country',

             title='Middle East Fatalities case', text='Fatalities', height=1000, orientation='h')

fig.show()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_data = list(train['Country'].unique())

country_geo = list(world['name'])



country_diff = [country for country in country_data if country not in country_geo]

country_diff
mapped = world.set_index('name').join(Country.set_index('Country')).reset_index()



to_be_mapped = 'Confirmed'

vmin, vmax = 0,1300000

fig, ax = plt.subplots(1, figsize=(25,25))



mapped.dropna().plot(column=to_be_mapped, cmap='Oranges', linewidth=0.8, ax=ax, edgecolors='0.8')

ax.set_title('Number of Confirmed  in countries', fontdict={'fontsize':30})

ax.set_axis_off()



sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []



cbar = fig.colorbar(sm, orientation='horizontal')
mapped = world.set_index('name').join(Country.set_index('Country')).reset_index()



to_be_mapped = 'Fatalities'

vmin, vmax = 0,1300000

fig, ax = plt.subplots(1, figsize=(25,25))



mapped.dropna().plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')

ax.set_title('Number of Fatalities  in countries', fontdict={'fontsize':30})

ax.set_axis_off()



sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []



cbar = fig.colorbar(sm, orientation='horizontal')
train['Province'] = train['Province'].fillna('')

temp = train[[col for col in train.columns if col != 'Province']]



europe = train[train['Date'] == max(train['Date'])].reset_index()

europe_ = europe.groupby('Country')['Confirmed', 'Fatalities'].sum().reset_index()
fig = px.choropleth(europe_, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Confirmed", range_color=[1,5000], 

                    color_continuous_scale="peach", 

                    title=' Confirmed Cases in the world')

fig.show()



fig = px.choropleth(europe_, locations="Country", 

                    locationmode='country names', color="Fatalities", 

                    hover_name="Fatalities", range_color=[1,5000], 

                    color_continuous_scale="peach", 

                    title=' Fatalities Cases in the world')

fig.show()
fig = px.choropleth(europe_, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country", range_color=[1,2000], 

                    color_continuous_scale='rainbow', 

                    title='Confirmed Cases in europe', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()



fig = px.choropleth(europe_, locations="Country", 

                    locationmode='country names', color="Fatalities", 

                    hover_name="Country", range_color=[1,2000], 

                    color_continuous_scale='rainbow', 

                    title='Fatalities Cases in europe', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
#bubble visualization

latitude = 36.0

longitude = 128.0

mapp = folium.Map(location=[latitude, longitude], zoom_start=12)

train_ = pd.DataFrame({

   'Province':list(train['Province']),

   'Lat':list(train['Lat']),

   'Long':list(train['Long']),

   'Confirmed':list(train['Confirmed']),

   'Fatalities':list(train['Fatalities'])

})
mapp = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(train_['Lat'], train_['Long'], train_['Confirmed'], train_['Province']):

    folium.CircleMarker([lat, lon],

                        radius=13,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Confirmed: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(mapp)

    folium.Map(titles='jj', attr="attribution")    

mapp
mapp = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(train_['Lat'], train_['Long'], train_['Fatalities'], train_['Province']):

    folium.CircleMarker([lat, lon],

                        radius=13,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Fatalities: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(mapp)

    folium.Map(titles='jj', attr="attribution")    

mapp