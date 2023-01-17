import pandas as pd
import random
import sklearn as sk
import os
import requests
import ephem
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bokeh.palettes import Viridis3, Viridis256,Magma256,Inferno256,Inferno,Category20_20
from sklearn.metrics import mean_absolute_error
import numpy as np
import bokeh.plotting as bkp
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.plotting import figure 
from bokeh.io import output_notebook,output_file,show
import bokeh.models as bkm
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    GeoJSONDataSource,
    LinearColorMapper,
    LogColorMapper,
    ColorBar,
    HoverTool,
    CrosshairTool,
    LinearAxis,
    Range1d,
)
output_notebook()
# Use ephem to get the distance in AU from earth to the moon on a given date.
# 1 AU = 150 million kilometers (93 million miles)
def moondist(s):
    dt = s
    n = ephem.Moon(dt)
    r = n.earth_distance
    return r
# Use ephem to get the Distance in AU from earth to Jupiter on a given date.
def jupiterdist(s):
    dt = s
    j = ephem.Jupiter(dt)
    r = j.earth_distance
    return r
# Use ephem to get the seperation in radians between the moon to Jupiter on a given date.
def jupitersep(s):
    dt = s
    j = ephem.Jupiter(dt)
    n = ephem.Moon(dt)
    r = ephem.separation(j, n)
    return r

np.random.seed(42)
# List the files
print(os.listdir("../input"))
# Loade Kaggle quake data into Pandas Dataframe
quake = pd.read_csv("../input/earthquake-database/database.csv",parse_dates=['Date'])
# Load NRC 10.7cm Solar Flux solar activity index into Pandas Dataframe.
solar = pd.read_csv("../input/102cm-solar-flux-data/solflux_monthly_average.txt",header=1,delim_whitespace=True,parse_dates = [['Year','Mon']])
# Rename one column of the DataFrame.
solar.rename(columns = {'Year_Mon':'Date'}, inplace = True)
solar.info()
quake.info()
# Break out ['Date'] to Month period.
quake['per'] = quake.Date.dt.to_period('M')
solar['per'] = solar.Date.dt.to_period('M')
# Perform a left join on Month period.
quake = pd.merge(quake, solar, on=['per'], how='left').drop('per', axis=1)
# Preview the first 20 rows of the dataframe
quake.head(20)
quake["Type"].value_counts()
quake["Magnitude Type"].value_counts()
quake["Magnitude Source"].value_counts()
quake["Location Source"].value_counts()
quake.describe()
# Augment with addtional astronomical data from ephem.

# Distance in Astronomical Units from earth to the moon at a given date.
quake['moondist'] = quake['Date_x'].apply(moondist)
# Distance of Jupiter from earth at a given Date
quake['jupiterdist'] = quake['Date_x'].apply(jupiterdist)
quake['jupitersep'] = quake['Date_x'].apply(jupitersep)

# Define Multiple y Labels grouped by Year. Mainly used for plotting
quake['yr'] = quake.Date_x.dt.year
grouped = quake.groupby("yr")
emag = grouped["Magnitude"]
avg = emag.mean()
std = emag.std()
maxmag = emag.max()
solarf = grouped["Adjflux"]
solarfmax = solarf.max()
moond = grouped["moondist"]
moonjup = grouped["jupitersep"]
moonjup_min = moonjup.min()
moond_avg = moond.mean()
moond_std = moond.std()
moond_min = moond.min()
jupd = grouped["jupiterdist"]
jupiter_avg = jupd.mean()
jupiter_std = jupd.std()
jupiter_min = jupd.min()
years = list(grouped.groups.keys())

# Plot Magnitude and standard deviation using bokeh.
source = ColumnDataSource(quake)
# Set up the Plot
p = figure(title="World Wide Earthquakes, magnitude >= 5.5",width=600, height=600,)
# Set up vert bars to denote std dev per year.
p.vbar(x=years, bottom=avg-std, top=avg+std, width=0.8, 
       fill_alpha=0.3, line_color=None, legend="Magnitude stddev")
# Set up to plot all 5.5 mag earthquakes and above per year.
p.circle(x="yr", y="Magnitude", size=5, alpha=0.5,
         color="firebrick",source=source)
p.legend.location = "top_left"
show(p)
# Inter active map with zoom, pan, and hover info for scatter markers. 

source = ColumnDataSource(quake)
# Define what tools to include in the toolbar.
tools = "pan,wheel_zoom,box_zoom,reset,previewsave,hover,crosshair"
# Open The Json World Map coordinates downloaded earlier to ./input
with open("../input/countriesgeojson/countries.geo.json", "r") as f:
    countries = bkm.GeoJSONDataSource(geojson=f.read())
# Set Up the Plot
p = bkp.figure(width=975, height=600, toolbar_location="above", title='World Countries',tools=tools, 
               x_axis_label='Longitude', y_axis_label='Latitude')
# Setup the world map
p.background_fill_color = "aqua"
p.x_range = bkm.Range1d(start=-180, end=180)
p.y_range = bkm.Range1d(start=-90, end=90)
p.patches("xs", "ys", color="white", line_color="black", source=countries)
# Setup color bar legend and color by magnitude for the actual cirlces.
mapper = LinearColorMapper(palette=Category20_20, low=5.5, high=10)
color_bar = ColorBar(color_mapper=mapper,label_standoff=3,title='Magnitude',location=(0, 0))
# The scatter markers
p.circle(
        x="Longitude", y="Latitude",
        fill_color={'field': 'Magnitude', 'transform': mapper}, size=4, alpha=1, line_color=None,
        source=source
)
# Where do we put the color bar?
p.add_layout(color_bar, 'left')
# Hovering over scatter marker (circle) will display the following:
hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Type", '@Type'),
    ("Year", '@yr'),
    ("Magnitude", '@Magnitude'),
    ("Moon Dist.(AU)", '@moondist'),
    ("Jupitor Dist.(AU)", '@jupiterdist'),
    ("(x,y)", "($x, $y)")
]    
show(p)
# Interactive Grid Plot.  Click on a data point on the MAX Magnitude plot to see the same time frame
#  such as solar flux.

# Lets give bokeh the defined y data points.
source = ColumnDataSource(data=dict(x=years, y1=jupiter_min,y2=moonjup_min,y3=maxmag,y4=solarfmax))
# What tools do we want present on the tool bar.
TOOLS = "box_select,lasso_select,hover,reset,help"
# Set up the plot to show the Distance between Jupitor
p1 = figure(tools=TOOLS,width=400, plot_height=375,title="Jupiter Dist (AU) from Earth")
p1.line('x', 'y1', alpha=0.5,color="black",source=source)
p1.circle('x', 'y1', size=7,alpha=0.5,color="firebrick",hover_color="red",source=source)
# Set up the plot to show the angular seperation in Radians.
p2 = figure(tools=TOOLS,width=400, plot_height=375,title="Jupiter/Moon Seperation")
p2.circle('x', 'y2', size=7,alpha=0.5,color="green",hover_color="red",source=source)
p2.line('x', 'y2', alpha=0.5,color="green",source=source)
# Set up the plot to show Max magnitude per year.
p3 = figure(tools=TOOLS,width=400, plot_height=375,title="Max Magnitude")
p3.line('x', 'y3', alpha=0.5,color="navy",source=source)
p3.circle('x', 'y3', size=7,alpha=0.5,color="navy",hover_color="red",source=source)
# Set up the plot to show 10.2cm (2.9Ghz) solar flux
p4 = figure(tools=TOOLS,width=400, plot_height=375,title="10.2cm Solar Flux")
p4.line('x', 'y4', alpha=0.5,color="orange",source=source)
p4.circle('x', 'y4', size=7,alpha=0.5,color="navy",hover_color="red",source=source)
# lets plot
p = gridplot([[p1, p2], [p3, p4]])
show(p)

