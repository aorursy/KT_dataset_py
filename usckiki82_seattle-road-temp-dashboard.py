# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

from pprint import pprint

import matplotlib.pyplot as plt

import itertools

from datetime import datetime, timedelta



# import plotly

from plotly import tools

import plotly.plotly as ply

import plotly.graph_objs as go

import folium as folium

from folium.plugins import MarkerCluster



from bokeh.plotting import figure, show, output_file

from bokeh.tile_providers import CARTODBPOSITRON



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Get data

stations = pd.read_csv("../input/road-weather-information-stations.csv")

stations.sort_values(by="DateTime")



# Fix Type Casting

stations['DateTime'] = pd.to_datetime(stations['DateTime'], format='%Y-%m-%dT%H:%M:%S')



stations.reset_index(inplace=True, drop=True)

# print(stations.columns)

# print(stations.head())
# make a dictionary of all stations



print(stations["StationName"].unique())

station_locs = stations[["StationName", "StationLocation"]].drop_duplicates()

station_dict = {}



for i, row in station_locs.iterrows():

    station_dict[row["StationName"]] = {} 

    loc_list = row["StationLocation"].replace("'", '').strip("{}").split(",")

    loc_dict = []

    

    for x in loc_list:

        loc_dict = loc_dict + x.strip().split(":")

        station_dict[row["StationName"]][loc_dict[0]]=0

    loc_dict = [str(x).strip() for x in loc_dict]

    station_dict[row["StationName"]] = dict(itertools.zip_longest(*[iter(loc_dict)] * 2, fillvalue=""))

    

# pprint(station_dict)
# Map with all locations

m = folium.Map(location=[47.6062, -122.3321], zoom_start=11, control_scale=True)



for station in station_dict:

    latest_data = stations[stations["StationName"] == station]

    latest_time = latest_data["DateTime"].max()

    latest_data = latest_data[latest_data["DateTime"] == latest_time].drop_duplicates()

    temp = np.float(latest_data["RoadSurfaceTemperature"])

    

    if  temp<= 32.0:

        c = 'red'

    elif temp <= 40.0:

        c = 'orange'

    else:

        c = 'green'



    m.add_child(folium.Marker(location = [np.float(station_dict[station]['latitude']),

                                           np.float(station_dict[station]['longitude'])], icon=folium.Icon(color=c, icon='circle')))

display(m)
# plot last air temp versus for last hour all stations

# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# create figure

# fig = tools.make_subplots(rows=len(stations["StationName"].unique()), cols=1)

plot_data = []

layout = dict(title = "Temperature - last 24hrs reported",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

N = 20

colors = plt.cm.gist_rainbow(np.linspace(0, 1, N))



# specify that we want a scatter plot with, with date on the x axis and meet on the y axis

for i,station in enumerate(stations["StationName"].unique()):

    data = stations[stations["StationName"] == station].sort_values(by="DateTime")

    last_time = data["DateTime"].max()

    first_time = last_time - timedelta(hours=24)

    data = data[(data["DateTime"] <= last_time) & (data["DateTime"] >= first_time)]

    rs_data = go.Scatter(x=data.DateTime, y=data.RoadSurfaceTemperature, name=f'{station} Road Surface Temp')

    at_data = go.Scatter(x=data.DateTime, y=data.AirTemperature, name=f'{station} Air Temp')

    plot_data = [rs_data, at_data]

# create and show our figure

fig = dict(data = plot_data, layout = layout)

iplot(fig)