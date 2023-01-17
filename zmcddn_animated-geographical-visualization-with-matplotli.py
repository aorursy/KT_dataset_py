# Import all packages

import sys

import os

from datetime import datetime

import random

import time

from math import sqrt

import itertools

import io

import base64



# Common DS libraries

import numpy as np

import pandas as pd

import scipy as sp

import sklearn



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.animation as animation

import seaborn as sns

import IPython

from IPython.display import display

from IPython.display import HTML



# Visualization Configurations

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



from pprint import pprint
print(f"Python version: {sys.version}")

print(f"pandas version: {pd.__version__}")
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df.describe()
df.info()
df.head(15)
df.sample(10)
fig, ax = plt.subplots(figsize=[26, 12])

sns.barplot(x=df['Province/State'], y=df['Recovered'], data=df[df['Country']=='Mainland China'], ax=ax)
import geopandas as gpd



# map plot based on country map

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_data = list(df['Country'].unique())

country_geo = list(world['name'])



country_diff = [country for country in country_data if country not in country_geo]

country_diff
df['Country'].replace(

    {

        'United States' : 'US', 

        'Mainland China': 'China',

        'Hong Kong' : 'China', 

        'Macau' : 'China',

    },

    inplace=True

)
import geopandas as gpd



death_sum = df.groupby('Country', sort=False)["Deaths"].sum().reset_index(name ='total_deaths')

death_sum = death_sum.sort_values(by="total_deaths", ascending=False)



mapped = world.set_index('name').join(death_sum.set_index('Country')).reset_index()



to_be_mapped = 'total_deaths'

vmin, vmax = 0, death_sum.total_deaths.iloc[0]

fig, ax = plt.subplots(1, figsize=(25,15))



mapped.dropna().plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')

ax.set_title('Number of deaths around the world', fontdict={'fontsize':30})

ax.set_axis_off()



sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []



cbar = fig.colorbar(sm, orientation='horizontal')
# Show the first and last entry datetime

df['Last Update'].iloc[0], df['Last Update'].iloc[-1]
# Change the datetime column type to datatime

df['Last Update'] =  pd.to_datetime(df['Last Update'])
def plot_animation(df=df, target_column='Recovered'):

    vmin, vmax = 0, df[target_column].max()

    iterations = df['Last Update'].nunique()



    fig = plt.figure(figsize=(26, 18)) 

    ax = fig.add_subplot(1,1,1)

    color_map = 'Wistia'



    def get_df_by_date(dof):

        start_date = str(df['Last Update'].iloc[0].date())

        end_date = df['Last Update'].unique()[dof-1]



        data = df.copy()

        data.loc[data['Last Update']>=end_date, target_column] = 0



        death_sum = data.groupby('Country', sort=False)[target_column].sum().reset_index(name=f'total_{target_column.lower()}')

        death_sum = death_sum.sort_values(by=f"total_{target_column.lower()}", ascending=False)



        mapped = world.set_index('name').join(death_sum.set_index('Country')).reset_index().dropna()

        return mapped



    def init():

        sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        sm._A = []



        cbar = fig.colorbar(sm, orientation='horizontal')



    def animate(frame_number):

        ax.clear()

        df_time = frame_number+1

        data = get_df_by_date(df_time)



        to_be_mapped = f'total_{target_column.lower()}'

        data.plot(column=to_be_mapped, cmap=color_map, linewidth=0.8, ax=ax, edgecolors='0.8')

        ax.set_title(f'Number of {target_column.lower()} around the world', fontdict={'fontsize':30})

        ax.set_axis_off()



    ani = animation.FuncAnimation(

        fig, animate, frames=list(range(1,iterations)), interval=25, repeat=False, blit=False, init_func=init

    )

    # plt.show()

    plt.close()

#     ani.save('animation.gif', writer='imagemagick', fps=1)

    return ani

    

ani = plot_animation()

# filename = 'animation.gif'

# video = io.open(filename, 'r+b').read()

# encoded = base64.b64encode(video)

# HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

HTML(ani.to_jshtml())
df['Province/State'].unique()
import argparse

import json

import requests

from urllib.request import urlopen



def get_province_coordinates(province_state_name):

    """get the coordinates for a given city/province/state name"""



    province_state_name = province_state_name.replace(' ', '+')

    

    base_url = 'https://nominatim.openstreetmap.org/search.php?'

    url = base_url + f'q={province_state_name}'



    with urlopen(url) as result:

        s = result.read()

        # Decode UTF-8 bytes to Unicode, and convert single quotes 

        # to double quotes to make it valid JSON

        # The coordinates data is stores in the first JSON data

        my_json = s.decode('utf8').replace("'", '"')

        api_payload = my_json[my_json.find('nominatim_results '):]

        api_payload = api_payload[:api_payload.find(';')]

        json_data = api_payload[api_payload.find('{'):]

        json_data = json_data[:json_data.find('}')+1]

#         print(api_payload)



        data = json.loads(json_data)

#         print(data['lat'], data['lon'])

        return data['lat'], data['lon']



def get_coordinate_map(show_result=False):

    """

    Get a dictionary of city coordinates

    Key is the city name, value is dict of lat and lon

    """

    

    city_dict = {}

    for city in df['Province/State'].dropna().unique():

        _lat, _lon = get_province_coordinates(city)

        if show_result:

            print(f'-- city: {city}')

            print(f'------ lat: {_lat}')

            print(f'------ lon: {_lon}\n')



        city_dict[city] = dict(lat=_lat, lon=_lon)

        

    return city_dict



print(get_coordinate_map()['New South Wales'])
df.groupby("Province/State")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
def get_city_numbers_map():

    _data = df.groupby("Province/State")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

    city_number_map = {}

    for city in _data["Province/State"]:

        city_number_map[city] = dict(

            Confirmed=_data.loc[_data['Province/State']==city]['Confirmed'],

            Deaths=_data.loc[_data['Province/State']==city]['Deaths'],

            Recovered=_data.loc[_data['Province/State']==city]['Recovered'],

        )

    return city_number_map
from mpl_toolkits.basemap import Basemap



CITY_COORS = get_coordinate_map()

CITY_NUM_TOTAL = get_city_numbers_map()

# target_column = 'Deaths'



def plot_city_animation(df=df, target_column='Confirmed'):

    iterations = df['Last Update'].nunique()



    fig = plt.figure(figsize=(26, 18)) 

    ax = fig.add_subplot(1,1,1)



    base_map = Basemap(projection='eck4',lon_0=0,resolution='c')

    base_map.drawcoastlines(linewidth=0.25)

    base_map.drawcountries(linewidth=0.25)

    base_map.drawstates(linewidth=0.25)

    base_map.drawmapboundary()

    base_map.fillcontinents(color='lightgray', zorder=1)

    scat = base_map.scatter([], [], ax=ax)

#     day_label = ax.text(-124,26,f"{df['Last Update'].iloc[0]}",fontsize=30)



    def get_data_this_time(dof):

        start_date = str(df['Last Update'].iloc[0].date())

        end_date = df['Last Update'].unique()[dof-1]



        data = df.copy()

        return data.loc[data['Last Update']<=end_date]



    def get_total(df, city, target_column):

        _data = df.groupby('Province/State').sum().reset_index()

        return _data.loc[_data['Province/State']==city][target_column]



    def get_scale_marker_size(number, old_max, scale_max):

        """Return a int number within the range 0 ~ scale_max for better display on map"""

        

        if number == 0:

            return 0

        

        if float(old_max) < scale_max:

            return number

        

        old_value = number

        new_max = 50

        new_min = 0

        old_min = 0

        new_value = (((old_value - old_min) * (new_max - old_min)) / (old_max - old_min)) + 0

        

        return 1 if 0<float(new_value)<1 else int(new_value)



    def animate(frame_number):

        df_time = frame_number+1

        data = get_data_this_time(df_time)



        for city in data['Province/State'].dropna().unique():

            x, y = base_map(CITY_COORS[city]['lon'], CITY_COORS[city]['lat'])

            total = int(get_total(data, city, target_column))

            city_max = CITY_NUM_TOTAL[city][target_column]

            

            marker_size = get_scale_marker_size(total, city_max, scale_max=50)

            

            base_map.plot(

                x,y,marker='o',color='Red',alpha=0.5,markersize=marker_size, ax=ax

            )



#             day_label.set_text(f"{df['Last Update'].iloc[df_time]}")

    #         plt.text(-124,26,f"{df['Last Update'].iloc[df_time]}",fontsize=20)



    ani = animation.FuncAnimation(fig, animate, frames=list(range(1,iterations)), interval=25, repeat=False, blit=False)

    plt.title(f'Number of {target_column.lower()} around the world', fontdict={'fontsize':30})

    # plt.show()

    plt.close()



    return ani

    

ani_city = plot_city_animation()

# filename = 'animation.gif'

# video = io.open(filename, 'r+b').read()

# encoded = base64.b64encode(video)

# HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

HTML(ani_city.to_jshtml())