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
#Import Libraries

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.style as style

style.use('seaborn-poster') #sets the size of the charts

style.use('ggplot')

plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M:%S"

pd.options.display.max_rows = 999

pd.set_option('display.max_columns', 500)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import geopandas as gpd

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

from shapely.geometry import Point, Polygon

import descartes

import math

from shapely.geometry import MultiPolygon

from fbprophet import Prophet
COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = gpd.read_file("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = gpd.read_file("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = gpd.read_file("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = gpd.read_file("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = gpd.read_file("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
geojson_confirmed = pd.read_json("../input/geojson/time_series_covid_19_confirmed.geojson")
COVID19_open_line_list.head(2)
COVID19_GeoDF = gpd.read_file("../input/covid19geojson/convertcsv.geojson")

crs = {'init' : 'epsg:4326'}

COVID19_GeoDF.head(2)
data = COVID19_GeoDF.loc[:, ["ID", "age", "sex", "city", "province", "country", "geometry"]].copy()
data["country"].unique()
china = data.loc[data['country'].isin(['China'])]

china.head()
ax = china.plot(figsize=(15,15), color='white', linestyle=':', edgecolor='blue')

data.plot(ax=ax, markersize=10)
data.geometry.head()
# Create a map

china_anhui = folium.Map(location=[31.646960, 117.716600], tiles='openstreetmap', zoom_start=10)



# Display the map

china_anhui
# Create a map

m = folium.Map(location=[31.646960, 117.716600], tiles='openstreetmap', zoom_start=2)



# Add points to the map

for idx, row in time_series_covid_19_confirmed.iterrows():

    Marker([row['Lat'], row['Long']], popup=row['Country/Region']).add_to(m)



# Display the map

m
# Create a map

m1 = folium.Map(location=[31.646960, 117.716600], tiles='openstreetmap', zoom_start=2)



# Add points to the map

for idx, row in time_series_covid_19_deaths.iterrows():

    Marker([row['Lat'], row['Long']], popup=row['Country/Region']).add_to(m1)



# Display the map

m1
# Create a map

m2 = folium.Map(location=[31.646960, 117.716600], tiles='openstreetmap', zoom_start=2)



# Add points to the map

for idx, row in time_series_covid_19_recovered.iterrows():

    Marker([row['Lat'], row['Long']], popup=row['Country/Region']).add_to(m2)



# Display the map

m2
import pandas as pd

COV19_time_confirmed = pd.read_csv("../input/cov19-time-series/COV19_time_confirmed.csv")

COV19_time_deaths = pd.read_csv("../input/cov19-time-series/COV19_time_deaths.csv")

COV19_time_recovered = pd.read_csv("../input/cov19-time-series/COV19_time_recovered.csv")
m = Prophet()

m.fit(COV19_time_confirmed)
future = m.make_future_dataframe(periods=30)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
m1 = Prophet()

m1.fit(COV19_time_deaths)
future = m1.make_future_dataframe(periods=30)

future.tail()
forecast = m1.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m1.plot(forecast)
fig2 = m1.plot_components(forecast)
fig1 = plot_plotly(m1, forecast)  # This returns a plotly Figure

py.iplot(fig)
m2 = Prophet()

m2.fit(COV19_time_recovered)
future = m2.make_future_dataframe(periods=30)

future.tail()
forecast = m2.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m2.plot(forecast)
fig2 = m2.plot_components(forecast)
fig = plot_plotly(m2, forecast)  # This returns a plotly Figure

py.iplot(fig)