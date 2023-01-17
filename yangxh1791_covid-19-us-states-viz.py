%matplotlib inline

import pandas as pd

import folium

import numpy as np

import warnings

import os

import geopandas

import branca

import json

from folium.plugins import Search



warnings.filterwarnings('ignore')



# Read Covid-19 US states summary data from NY times' repo.

url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master'

covid_us = f'{url}/us-states.csv'

df = pd.read_csv(covid_us)

df.date = pd.to_datetime(df.date)

df.set_index('date',inplace = True)

# US states geojson data

state_json = os.path.join('/kaggle/input/usa-states','usa-states.json')

print("Covid-19 data up to the date: ",df.index.max())
#

# Generate an interactive heatmap of Covid-19 accumulated cases & deaths for US states.

# 

# Inputs:

#       df        : dataframe containing covid-19 data.

#       date      : Date of covid-19 data to be used.

#       state_json: Path of US state geojson file.

# Returns:

#       folium map

#

def interactive_heatmap(df,date,state_json):

    # Filter the dataframe 'df' with 'date'

    df = df.loc[df.index==date]

    with open(state_json, 'r') as f:

        a = json.load(f)

    #

    # Use dict 'geojson' to store US states geojson data combined with Covid-19 data

    # so that it can be passed to 'folium.GeoJson' for look-up.

    #

    geojson = {'type': 'FeatureCollection', 'features': []}

    for feature in list(a['features']):

        state = feature['properties']['name']

        if state in df.state.values:

            feature['properties']['cases'] = int(df[df.state==state]['cases'].values[0])

            feature['properties']['deaths'] = int(df[df.state==state]['deaths'].values[0])

        else:

            feature['properties']['cases'] = int(0)

            feature['properties']['deaths'] = int(0)

        geojson['features'].append(feature)

    #

    # Create a colormap

    #

    vmin, vmax = df['cases'].quantile([0.025,0.975]).apply(lambda x: round(x, 0))

    colormap = branca.colormap.linear.YlGn_09.scale(vmin,vmax)

    colormap.caption="Covid-19 cases in the U.S., {}".format(date)



    m = folium.Map(location=[38,-97], zoom_start=4)

    style_function = lambda x: {

        'fillColor': colormap(x['properties']['cases']),

        'color': 'black',

        'weight':2,

        'fillOpacity':0.5

    }



    stategeo = folium.GeoJson(

        geojson,

        name='US States',

        style_function=style_function,

        tooltip=folium.GeoJsonTooltip(

            fields=['name', 'cases','deaths'],

            aliases=['State', 'Cases','Deaths'], 

            localize=True

        )

    ).add_to(m)

    folium.LayerControl().add_to(m)

    colormap.add_to(m)

    statesearch = Search(

        layer=stategeo,

        geom_type='Polygon',

        placeholder='Search for a US State',

        collapsed=False,

        search_label='name',

        weight=3).add_to(m)

    return m
interactive_heatmap(df,'2020-3-5',state_json)
map = interactive_heatmap(df,df.index.max(),state_json)

map
map.save('index.html')
'''

states = geopandas.read_file(state_json,

    driver='GeoJSON'

)



def plot_heatmap(df):

    state_geo = 'us-states.json'

    m = folium.Map(location=[48, -102], zoom_start=3)



    folium.Choropleth(

        geo_data=state_geo,

        name='choropleth',

        data=df,

        columns=['state', 'log_cases'],

        key_on='feature.properties.name',

        fill_color='YlGn',

        fill_opacity=0.7,

        line_opacity=0.2,

        bins=8,

        legend_name='Covid-19 cases (log)'

    ).add_to(m)



    folium.LayerControl().add_to(m)

    return m

'''


