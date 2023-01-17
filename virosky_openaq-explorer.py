# utility

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

import panel as pn

pn.extension()



#bigquery

import bq_helper

from google.cloud import bigquery

from bq_helper import BigQueryHelper



#folium

import folium

from folium import plugins

from folium.plugins import HeatMap

import bq_helper



import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import pycountry
client = bigquery.Client()



# Helper object

openAQ = bq_helper.BigQueryHelper(active_project='bigquery-public-data',

                                 dataset_name='openaq')

# List of table

openAQ.list_tables()
query1 = """

        SELECT DISTINCT 

        pollutant, 

        value,

        latitude,

        longitude,

        city,

        FROM `bigquery-public-data.openaq.global_air_quality` 

        WHERE timestamp >'2020-01-01'

        """
world_map = openAQ.query_to_pandas_safe(query1)

world_map.head(3)
pollutant_level=world_map[world_map.pollutant=='co']

pollutant=pollutant_level[['latitude','longitude','value']]

m = folium.Map([42.50, 12.50], tiles='CartoDB Positron', zoom_start=2,zoom_control=True,

               scrollWheelZoom=False,

               dragging=True)

HeatMap(pollutant, min_opacity=0.4).add_to(m)



folium.LayerControl().add_to(m)



HeatMap(pollutant,radius=0.001).add_to(folium.FeatureGroup(name='Heat Map').add_to(m))

folium.LayerControl().add_to(m)



loc = 'Carbon Monoxide(CO) Level in 2020'

title_html = '''<h3 align="right" style="font-size:1; font-family:'Helvetica'"><b>{}</b></h3>

             '''.format(loc)   



m.get_root().html.add_child(folium.Element(title_html))

m
query2 = """

        SELECT DISTINCT 

        pollutant, 

        value,

        country,

        EXTRACT(YEAR FROM timestamp) as year,

        EXTRACT(MONTH FROM timestamp) as month

        FROM `bigquery-public-data.openaq.global_air_quality` 

        WHERE country='IT' and timestamp > '2019-01-01'

        """
countries= openAQ.query_to_pandas_safe(query2)
plt.figure(figsize=(15,7))

sns.barplot(data=countries,x='year',y='value',hue='pollutant',edgecolor='black',linewidth=1)

plt.title('Italy - Average pollutants value between 2019-2020',size=17)

plt.xlabel("")

plt.show()
it_poll = countries[countries.year==2020]

# Plot 2020

plt.figure(figsize=(17,5))

ax=sns.barplot(data=it_poll,x='month',y='value',hue='pollutant',edgecolor='black')

ax.set_xticklabels(['January','February','March','April','May','June','July'])

plt.title('Italy- Average pollutants value during 2020',size=17)

plt.xlabel('')

plt.show()
query3 = """

        SELECT DISTINCT 

        pollutant, 

        value,

        country,

        city,

        EXTRACT(YEAR FROM timestamp) as year,

        EXTRACT(MONTH FROM timestamp) as month

        FROM `bigquery-public-data.openaq.global_air_quality` 

        WHERE country='IT' and timestamp > '2020-01-01'

        """
it_cities= openAQ.query_to_pandas_safe(query3)
it_cities=it_cities.groupby(['pollutant','city'])['value'].mean().reset_index()
poll = it_cities["pollutant"].unique()

fig, axes = plt.subplots(1,len(poll),figsize=(17,5)) 



for pol, ax in zip(poll, axes):

    it_cities[it_cities.pollutant==pol].groupby('city')['value'].mean().reset_index().sort_values(by='value',ascending=False)[:5].plot(kind='bar',x='city',edgecolor='black',ax=ax)    

    ax.set_title(pol)

    ax.legend('')

    ax.set_xlabel('',rotation=90)

    plt.tight_layout()
