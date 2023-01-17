# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows =10

import bq_helper

# google bigquery library for quering data
from google.cloud import bigquery

# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins as plugins

%matplotlib inline
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
#Schema 
open_aq.table_schema('global_air_quality')
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
query_aqi = """
            SELECT EXTRACT(YEAR FROM timestamp) as `Year`,
                   AVG(value) as `Average`,
                   latitude,
                   longitude , city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'µg/m³'AND country = 'IN'
        GROUP BY Year, 
                 latitude,
                 longitude , city
        """
aqi = open_aq.query_to_pandas_safe(query_aqi)

aqi.head()
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
fig = plt.figure(figsize=(14,10))
plt.style.use('ggplot')

def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('AQI in Year: '+str(Year))
    m4 = Basemap(llcrnrlat=4, urcrnrlat=35, llcrnrlon=65,urcrnrlon=95,projection='cyl')
    m4.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m4.fillcontinents(color='grey', alpha=0.3)
    m4.drawcoastlines(linewidth=0.1, color="white")
    m4.shadedrelief()
    
    lat_y = list(aqi[aqi['Year'] == Year]['latitude'])
    lon_y = list(aqi[aqi['Year'] == Year]['longitude'])
    lat,lon = m4(lat_y,lon_y) 
    avg = np.log(aqi[aqi['Year'] == Year]['Average'])
    m4.scatter(lon,lat,c = avg,lw=2, alpha=0.3,cmap='hot_r')
   
ani = animation.FuncAnimation(fig,animate,list(aqi['Year'].unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
query = """SELECT city,COUNT(city) as `count`
    FROM `bigquery-public-data.openaq.global_air_quality`
    where country = 'IN'
    GROUP BY city
    HAVING COUNT(city) >10
    ORDER BY `count` DESC
    
    """
cnt = open_aq.query_to_pandas_safe(query)

cnt.head()
plt.style.use('bmh')
plt.figure(figsize=(15,5))
sns.barplot(cnt['city'], cnt['count'], palette='magma')
plt.xticks(rotation=45)
plt.title('Distribution of states listed in data');
query = """SELECT city,latitude,longitude,averaged_over_in_hours,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND country = 'IN'
            GROUP BY latitude,city,longitude,averaged_over_in_hours   
            ORDER BY Average DESC
            """
location = open_aq.query_to_pandas_safe(query)
location.dropna(axis=0, inplace=True)
location.head(10)
plt.style.use('ggplot')
f,ax = plt.subplots(figsize=(15,10))
m1 = Basemap(projection='cyl',llcrnrlat=4, urcrnrlat=35, llcrnrlon=65, urcrnrlon=95,
            resolution='c',lat_ts=True)

m1.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m1.fillcontinents(color='grey', alpha=0.3)
m1.drawcoastlines(linewidth=0.1, color="blue")
m1.shadedrelief()

hour = location['averaged_over_in_hours']
avg = np.log(location['Average'])
m1loc = m1(location['latitude'].tolist(),location['longitude'])
m1.scatter(m1loc[1],m1loc[0],lw=3,alpha=1,cmap='hot',\
          c=avg,s=hour)
plt.title('Average Air qulity index in unit $ug/m^3$ value')
plt.colorbar(label=' Average Log AQI value in unit $ug/m^3$');
#INDIA location
query = """SELECT 
            MAX(latitude) as `max_lat`,
            MIN(latitude) as `min_lat`,
            MAX(longitude) as `max_lon`,
            MIN(longitude) as `min_lon`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' """
in_loc = open_aq.query_to_pandas_safe(query)
in_loc
query = """ SELECT city,latitude,longitude,averaged_over_in_hours,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' AND unit = 'µg/m³'
            GROUP BY latitude,city,longitude,averaged_over_in_hours,country """
in_aqi = open_aq.query_to_pandas_safe(query)
in_aqi
# INDIA
min_lat = in_loc['min_lat']
max_lat = in_loc['max_lat']
min_lon = in_loc['min_lon']
max_lon = in_loc['max_lon']

plt.figure(figsize=(15,10))
m3 = Basemap(projection='cyl',llcrnrlat=4, urcrnrlat=35, llcrnrlon=65, urcrnrlon=95,
            resolution='c',lat_ts=True)
m3.drawcounties()
m3.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m3.fillcontinents(color='grey', alpha=0.3)
m3.drawcoastlines(linewidth=0.1, color="white")
m3.drawstates()
avg = np.log((in_aqi['Average']))
h = in_aqi['averaged_over_in_hours']
m3loc = m3(in_aqi['latitude'].tolist(),in_aqi['longitude'])
m3.scatter(m3loc[1],m3loc[0],s = h,c = avg,lw=3,alpha=1,cmap='hot')
plt.colorbar(label = 'Average Log AQI value in unit $ug/m^3$')
plt.title('Average Air qulity index in unit $ug/m^3$ of India');
query_source = """
                    SELECT 
                    DISTINCT source_name, 
                             country 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    where country = 'IN'
                    ORDER BY source_name
                    """

source = open_aq.query_to_pandas_safe(query_source)

source.head(10)
query_country_count_per_source = """
                    SELECT 
                    DISTINCT source_name,
                    COUNT(city) AS CITY_Count
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    where country ='IN'
                    GROUP BY source_name , city
                    
                    ORDER BY CITY_Count DESC
                    """
# Country_Count > 50 (Ignoring the negligible values)
country_count_per_source = open_aq.query_to_pandas_safe(query_country_count_per_source)

country_count_per_source.head(10)
plt.subplots(figsize=(12,10))
sns.barplot(x='CITY_Count',y='source_name',data=country_count_per_source,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Source Name', fontsize=15)
plt.xticks(rotation=45,fontsize=10)
plt.xlabel('City Count', fontsize=15)
plt.title('Sources per city Count', fontsize=24)
plt.savefig('sources_per_city_count.png')
plt.show()
query_pollutants = """
                    SELECT 
                    DISTINCT pollutant, unit
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE source_name IN ('AirNow', 'Anand Vihar', 'CPCB', 'Spartan', 'caaqm', 'data.gov.in') AND country = 'IN'
                    ORDER BY pollutant
                    """

pollutants = open_aq.query_to_pandas_safe(query_pollutants)

pollutants.head(15)
query_co = """
    SELECT city, 
           avg(value) as Avg_Value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'co'
      AND unit = 'µg/m³' AND country = 'IN'
      GROUP BY city, source_name
      ORDER BY Avg_Value ASC
        """

co = open_aq.query_to_pandas_safe(query_co)

co.head(10)
plt.subplots(figsize=(15,6))

sns.barplot(x='city',y='Avg_Value',data=co,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Carbon Monoxide Gas values in µg/m³', fontsize=20)
plt.xticks(rotation=90,fontsize=5)
plt.xlabel('city', fontsize=20)
plt.title('Average value of Carbon Monoxide gas in different cities', fontsize=24)
plt.savefig('avg_co.png')
plt.show()
query_no2 = """
    SELECT city, 
           avg(value) as Avg_Value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'no2'
      AND unit = 'µg/m³' AND country = 'IN'
      GROUP BY city, source_name
      ORDER BY Avg_Value ASC
        """

no2 = open_aq.query_to_pandas_safe(query_no2)

no2.head(5)
plt.subplots(figsize=(15,6))
sns.barplot(x='city',y='Avg_Value',data=no2,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Nitrogen Dioxide Gas values in µg/m³', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('city', fontsize=20)
plt.title('Average value of Nitrogen Dinoxide gas in different cities', fontsize=24)
plt.savefig('avg_no2.png')
plt.show()
query_o3 = """
    SELECT city, 
           avg(value) as Avg_Value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'o3'
      AND unit = 'µg/m³' AND country = 'IN'
      GROUP BY city, source_name
      ORDER BY Avg_Value ASC
        """

o3 = open_aq.query_to_pandas_safe(query_o3)

o3.head(10)
plt.subplots(figsize=(15,6))
sns.barplot(x='city',y='Avg_Value',data=o3,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Ozone Gas values in µg/m³', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('City', fontsize=20)
plt.title('Average value of Ozone gas in different cities', fontsize=24)
plt.savefig('avg_o3.png')
plt.show()
