# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly

import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True) 
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the dataset into pandas dataframe
dataset = pd.read_csv("../input/countries of the world.csv")
#Finding out the number of rows and columns in our dataset i.e total countries and attributes
print(dataset.shape)
dataset.columns = ['country', 'region', 'population', 'area','population_density', 'coastline','migration', 'infant_mortality',
       'gdp', 'literacy', 'phones_per_1000', 'arable','crops', 'other', 'climate', 'birthrate', 'deathrate',
       'agriculture', 'industry', 'service']
print(dataset.info())
print(dataset.dtypes)
dataset.head()
column_list =['population_density', 'coastline','migration', 'infant_mortality',
      'literacy', 'phones_per_1000', 'arable','crops', 'other', 'climate', 'birthrate', 'deathrate','agriculture', 'industry', 'service']
for item in column_list:
    def column_data(item):
        dataset[item]= dataset[item].str.replace(',' ,'.').astype(float) 
    column_data(item) 

dataset.country = dataset.country.astype('category')
dataset.region = dataset.region.str.strip().astype('category')    

dataset.head()   
dataset.fillna(dataset.mean(),inplace=True)
dataset.head()
# We need to create data object and layout object  which contains a dict
# colorscale → This is the color for the geographical map elements
# locations → This is the data for the state abbreviations
# locationmode → This lets plotly know we what nation to use
# z → This is the numerical measurement for each state element; This should be of the same index sequence as the locations argument
# text → This is the categorical value for each element
# colorbar → Title for right side bar

data = dict(type='choropleth',
                locations = dataset['country'],
                locationmode = 'country names',
                z = dataset['population']/1000,
                text = dataset['country'],
                colorbar = {'title':'Population Scale'},
                colorscale = 'Viridis',
                reversescale = True
                )

# Lets make a layout
layout = dict(title='Population Spread Across The Globe ',
geo = dict(showframe=False,projection={'type':'natural earth'}))

worldmap = go.Figure(data = [data],layout = layout)
plotly.offline.iplot(worldmap, validate=True)
#Another set of relationships can be gauged using bubble plots
axis0='literacy'
axis1='infant_mortality'
trace_items = []
for item in list(dataset['region'].unique().astype(str)):
    trace_item = go.Scatter(
    x = dataset[axis0][dataset['region'] == item],
    y =  dataset[axis1][dataset['region'] == item],
    mode='markers',
    name= item,
    text=dataset['country'][dataset['region'] == item],
    marker=dict(
    size=list(np.cbrt(dataset['population'][dataset['region'] == item])/10)
    ))
    trace_items.append(trace_item)



data = trace_items

layout = go.Layout(
    title= 'literacy vs Infant_Mortality',
    xaxis=dict(
        title=' Literacy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18
        )
    ),
    yaxis=dict(
        title='Infant_Mortality',
        titlefont=dict(
            family='Courier New, monospace',
            size=18           
        )
    )
)

fig1 = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig1, show_link=True)
