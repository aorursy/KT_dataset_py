# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly
from plotly.offline import init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
import plotly.tools as tls

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

init_notebook_mode(connected=True)

# Any results you write to the current directory are saved as output.
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")

table = ff.create_table(df)
iplot(table)
# Define figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}
figure['data'] = [go.Bar(x = df.School,y = df.Women)]
figure['layout'] = go.Layout(title = 'Women Earnings')
iplot(figure)
"""
matplotlib.pyplot charts can be displayed through pyplot 
"""
x = df.School
y = df.Women
plt.plot(x,y)
plt.title('Women Earnings')
plt.tight_layout()

fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly( fig )
iplot(plotly_fig)
trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = [trace0, trace1]

iplot(data)
y0 = np.random.randn(50)-1
y1 = np.random.randn(50)+1

trace0 = go.Box(
    y=y0,
    name = 'Sample 1',
    marker = dict(
        color = 'rgb(255, 55, 55)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Sample 2',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
trace0 = go.Box(
    y=[2.37, 2.16, 4.82, 1.73, 1.04, 0.23, 1.32, 2.91, 0.11, 4.51, 0.51, 3.75, 1.35, 2.98, 4.50, 0.18, 4.66, 1.30, 2.06, 1.19],
    name='Only Mean',
    marker=dict(
        color='rgb(8, 81, 156)',
    ),
    boxmean=True
)
trace1 = go.Box(
    y=[2.37, 0.16, 4.82, 1.73, 1.04, 0.23, 1.32, 2.91, 0.11, 4.51, 0.51, 3.75, 1.35, 2.98, 4.50, 0.18, 4.66, 1.30, 2.06, 1.19],
    name='Mean & SD',
    marker=dict(
        color='rgb(10, 140, 208)',
    ),
    boxmean='sd'
)
data = [trace0, trace1]
iplot(data)
x = ['day 1', 'day 1', 'day 1', 'day 1', 'day 1', 'day 1',
     'day 2', 'day 2', 'day 2', 'day 2', 'day 2', 'day 2']

trace0 = go.Box(
    y=[0.2, 0.2, 0.6, 1.0, 0.5, 0.4, 0.2, 0.7, 0.9, 0.1, 0.5, 0.3],
    x=x,
    name='kale',
    marker=dict(
        color='#3D9970'
    )
)
trace1 = go.Box(
    y=[0.6, 0.7, 0.3, 0.6, 0.0, 0.5, 0.7, 0.9, 0.5, 0.8, 0.7, 0.2],
    x=x,
    name='radishes',
    marker=dict(
        color='#FF4136'
    )
)
trace2 = go.Box(
    y=[0.1, 0.3, 0.1, 0.9, 0.6, 0.6, 0.9, 1.0, 0.3, 0.6, 0.8, 0.5],
    x=x,
    name='carrots',
    marker=dict(
        color='#FF851B'
    )
)
data = [trace0, trace1, trace2]
layout = go.Layout(
    yaxis=dict(
        title='normalized moisture',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
import random
import plotly.plotly as py

from numpy import *

N = 30     # Number of boxes

# generate an array of rainbow colors by fixing the saturation and lightness of the HSL representation of colour 
# and marching around the hue. 
# Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, N)]

# Each box is represented by a dict that contains the data, the type, and the colour. 
# Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:
data = [{
    'y': 3.5*sin(pi * i/N) + i/N+(1.5+0.5*cos(pi*i/N))*random.rand(10), 
    'type':'box',
    'marker':{'color': c[i]}
    } for i in range(int(N))]

# format the layout
layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False},
          'yaxis': {'zeroline':False,'gridcolor':'white'},
          'paper_bgcolor': 'rgb(233,233,233)',
          'plot_bgcolor': 'rgb(233,233,233)',
          }

iplot(data)
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
df.head()

df['text'] = df['name'] + '<br>Population ' + (df['pop']/1e6).astype(str)+' million'
limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgrey"]
cities = []
scale = 5000

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['pop']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    cities.append(city)

layout = dict(
        title = '2014 US city populations<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
iplot( fig, validate=False)
# read in volcano database data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv',\
                encoding='iso8859-1')

# frequency of Country
freq = df

freq = freq.Country.value_counts().reset_index().rename(columns={'index': 'x'})

# plot(1) top 10 countries by total volcanoes
locations = go.Bar(x=freq['x'][0:10],y=freq['Country'][0:10], marker=dict(color='#CF1020'))

# read in 3d volcano surface data
df_v = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv')

# plot(2) 3d surface of volcano
threed = go.Surface(z=df_v.values.tolist(), colorscale='Reds', showscale=False)

# plot(3)  scattergeo map of volcano locations

trace3 = {
  "geo": "geo3",
  "lon": df['Longitude'],
  "lat": df['Latitude'],
  "hoverinfo": 'text',
  "marker": {
    "size": 4,
    "opacity": 0.8,
    "color": '#CF1020',
    "colorscale": 'Viridis'
  },
  "mode": "markers",
  "type": "scattergeo"
}

data = [locations, threed, trace3]

# control the subplot below using domain in 'geo', 'scene', and 'axis'
layout = {
  "plot_bgcolor": 'black',
  "paper_bgcolor": 'black',
  "titlefont": {
      "size": 20,
      "family": "Raleway"
  },
  "font": {
      "color": 'white'
  },
  "dragmode": "zoom",
  "geo3": {
    "domain": {
      "x": [0, 0.55],
      "y": [0, 0.9]
    },
    "lakecolor": "rgba(127,205,255,1)",
    "oceancolor": "rgb(6,66,115)",
    "landcolor": 'white',
    "projection": {"type": "orthographic"},
    "scope": "world",
    "showlakes": True,
    "showocean": True,
    "showland": True,
    "bgcolor": 'black'
  },
  "margin": {
    "r": 10,
    "t": 25,
    "b": 40,
    "l": 60
  },
  "scene": {"domain": {
      "x": [0.5, 1],
      "y": [0, 0.55]
    },
           "xaxis": {"gridcolor": 'white'},
           "yaxis": {"gridcolor": 'white'},
           "zaxis": {"gridcolor": 'white'}
           },
  "showlegend": False,
  "title": "<br>Volcano Database",
  "xaxis": {
    "anchor": "y",
    "domain": [0.6, 0.95]
  },
  "yaxis": {
    "anchor": "x",
    "domain": [0.65, 0.95],
    "showgrid": False
  }
}

annotations = { "text": "Source: NOAA",
               "showarrow": False,
               "xref": "paper",
               "yref": "paper",
               "x": 0,
               "y": 0}

layout['annotations'] = [annotations]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Mixed Subplots Volcano")

