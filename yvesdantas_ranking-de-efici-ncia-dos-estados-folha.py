import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')
data = pd.read_excel('../input/reef.xlsx')

data.head()
#normalizing data

min_max_scaler = preprocessing.MinMaxScaler()

features = min_max_scaler.fit_transform(np.array(data.drop(['Estado'], axis=1)))

pd.DataFrame(features).head()
#kmeans application

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(features)

#organizing data

data['Clusters'] = kmeans.labels_

def converter(change):
    if change == 1:
        return 'Grupo 1'
    elif change == 2:
        return 'Grupo 2'
    else:
        return 'Grupo 3'

data['Clusters'] = data['Clusters'].apply(converter)

sns.lmplot(x='REE-F', y='Receita per capita', data=data, aspect=2, hue='Clusters', fit_reg=False)
#preparing data frame
a = data[data['Clusters']=='Grupo 1']
b = data[data['Clusters']=='Grupo 2']
c = data[data['Clusters']=='Grupo 3']

# Creating trace1
trace01 = go.Scatter(
    x = a['REE-F'],
    y = a['Receita per capita'],
    text = a['Estado'],
    name = 'Grupo 1',
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgba(16, 112, 2, 0.8)',
        line = dict(
            width = 1,
            
        )
    )
)

# Creating trace2
trace02 = go.Scatter(
    x = b['REE-F'],
    y = b['Receita per capita'],
    text = b['Estado'],
    name = 'Grupo 2',
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgba(80, 26, 80, 0.8)',
        line = dict(
            width = 1,
            
        )
    )
)

# Creating trace3
trace03 = go.Scatter(
    x = c['REE-F'],
    y = c['Receita per capita'],
    text = c['Estado'],
    name = 'Grupo 3',
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 1,
            
        )
    )
)

Data = [trace01, trace02, trace03]

layout = dict(title = 'Nota REE-F vs Receita per capita',
              yaxis = dict(zeroline = False),
              xaxis = dict(title= 'Ranking de Eficiência dos Estados - Folha', zeroline = False)
             )

fig = dict(data=Data, layout=layout)
iplot(fig, filename='styled-scatter')
#importing libraries for choropleth maps

import geopandas as gpd
import folium 
import os
import json
#loading data

state_geo = gpd.read_file(os.path.join('../input/States_IBGE.json'))
state_data = data[['Estado','REE-F']]
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(features)
state_data['Clusters'] = kmeans.labels_
state_data.head()
#kmeans choropleth map

m = folium.Map(location=[-15.7, -47.8], zoom_start=4)

m.choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['Estado', 'Clusters'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='kmeans results',
)

folium.LayerControl().add_to(m)

m
#

m = folium.Map(location=[-15.7, -47.8], zoom_start=4)

m.choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['Estado', 'REE-F'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='REE-F score',
)

folium.LayerControl().add_to(m)

m