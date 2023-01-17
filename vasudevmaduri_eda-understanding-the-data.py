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



import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

import math

import json

import numpy as np

import pandas as pd

import networkx as nx

import cartopy.crs as ccrs

import matplotlib.pyplot as plt

from IPython.display import Image

import missingno as msno

import seaborn as sns



%matplotlib inline

# Any results you write to the current directory are saved as output.
df20 = pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv")

df19 = pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv")
#Parsing Dates

df20['Date'] = pd.to_datetime('2020-01-' + df20['DAY_OF_MONTH'].apply(str))

df20['day_name']=df20['Date'].dt.weekday_name
df20.head(5)
df20.columns
#prepare the data

df_carieer = pd.DataFrame(df20['OP_CARRIER'].value_counts().reset_index().values, columns=["OP_CARRIER", "AggregateOP"])

# df_carieer = df_carieer.sort_index(axis = 0, ascending=True)

df_carieer= df_carieer.sort_values('AggregateOP',ascending=False)



fig = px.bar(df_carieer, y='AggregateOP', x='OP_CARRIER', text='AggregateOP', opacity = 0.8)

fig.update_traces(texttemplate='%{text:.1s}', textposition='outside')

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_x = 0.5,title_text='Carriers Count (OP Carrier)', yaxis=dict(title='Count'),

                  xaxis=dict(title='OP Carrier Code'))

fig.add_annotation( x='WN', y=100000, text="Highest OP CARRIER - WN",showarrow=True, font=dict( family="Courier New, monospace", size=10, color="#ffffff" ), align="right", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363", ax=120, ay=0, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ff7f0e", opacity=0.8)



fig.add_trace(go.Scatter(x=df_carieer['OP_CARRIER'], y=df_carieer['AggregateOP'],

                    mode='lines+markers',opacity = 0.3,showlegend=False,

                   line = dict(

        smoothing = 1.2, color = 'blue',

        shape = "spline"

    )))



fig.show()



# df20.columns

df_Org = pd.DataFrame(df20['ORIGIN'].value_counts().reset_index().values, columns=["ORIGIN", "AggregateOrigin"])



df_Org = df_Org.sort_values('AggregateOrigin',ascending=False).head(10)

df_Org = df_Org.sort_values('AggregateOrigin',ascending=True)



fig = px.bar(df_Org, x='AggregateOrigin', y='ORIGIN', text='AggregateOrigin', orientation ='h',opacity = 0.8)

fig.update_traces(texttemplate='%{text:.5s}', textposition='outside')

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_text='Top 10 Airports w.r.t Arrivals', title_x = 0.5)

fig.add_annotation( x= 30000, y='ATL', text="World's Busiest Airport <br> <b>Hartsfield–Jackson Atlanta International Airport<b>",showarrow=True, font=dict( family="Arial", size=12, color="#ffffff" ), align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363", ax=-30, ay=80, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#32a848", opacity=0.8)



fig.show()

#Busiest Airport Source :https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States


df_dest = pd.DataFrame(df20['DEST'].value_counts().reset_index().values, columns=["DEST", "AggregateDest"])



df_dest = df_dest.sort_values('AggregateDest',ascending=False).head(10)

df_dest = df_dest.sort_values('AggregateDest',ascending=True)







fig = px.bar(df_dest, x='AggregateDest', y='DEST', text='AggregateDest', orientation ='h',opacity = 0.8)

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_text='Top 10 Airports w.r.t Departures', title_x = 0.5)

fig.add_annotation( x= 30000, y='ATL', text="World's Busiest Airport <br> <b>Hartsfield–Jackson Atlanta International Airport<b>",showarrow=True, font=dict( family="Arial", size=12, color="#ffffff" ), align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363", ax=-30, ay=80, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#32a848", opacity=0.8)

fig.show()



#Busiest Airport Source :https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States
#Comparison of both origin & desination flights

fig = go.Figure()

# Create and style traces

fig.add_trace(go.Scatter(x=df_dest['DEST'], y=df_dest['AggregateDest'], name='Destination Airport',

                         line=dict(color='firebrick', width=2)))

fig.add_trace(go.Scatter(x=df_Org['ORIGIN'], y=df_Org['AggregateOrigin'], name = 'Origin Airport',

                         line=dict(color='royalblue', width=1)))

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_text='Top 10 Airports w.r.t Arrivals & Destinations', title_x = 0.5)

fig.update_layout( yaxis=dict(title='Count'),xaxis=dict(title='IATA Code'))
#comparing arrivals and departures time frames



values = [df20['DEP_DEL15'].value_counts()[0],df20['DEP_DEL15'].value_counts()[1]]

labels = ["Delayed (<15 minutes)", "Delayed (>15 minutes)"]

colors = ['lightgreen','red']

values_arr = [df20['ARR_DEL15'].value_counts()[0],df20['ARR_DEL15'].value_counts()[1]]



fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                    subplot_titles=['Departures', 'Arrivals'])

fig.add_trace(go.Pie(labels=labels, values=values, pull=[0, 0.1],textinfo = 'label+percent'),1,1)

fig.add_trace(go.Pie(labels=labels, values=values_arr, pull=[0, 0.1],textinfo = 'label+percent'),1,2)

fig.update_traces( textinfo='value', textfont_size=14,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_text='Flights Delayed', title_x = 0.5, legend_title='<b>               Flights</b>',legend=dict(x=.45, y=0.6))





fig.show()
df_time = pd.DataFrame(df20['DEP_TIME_BLK'].value_counts().reset_index().values, columns=["DEP_TIME_BLK", "AggregateDepTime"])



df_time = df_time.sort_values('DEP_TIME_BLK',ascending=True)



width = [0.5] * 19

width[0] = 0.9

colors = ['#053752','#f29624','#f29624','#e5de44','#e5de44','#eae54b','#eae54b','#f5f259','#f5f259','#f5f259','#f8bd4c','#fbd063','#5595a9','#417c93','#2d647d','#1a4d68','#053752','#053752','#053752']





fig = go.Figure(data=[go.Bar(x = df_time['DEP_TIME_BLK'], y = df_time['AggregateDepTime'], width = width, marker_color =colors, opacity =0.8, marker_line_width=2, text = df_time['AggregateDepTime'],textposition='outside' )])



fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', title_x = 0.5,title_text='<b>Departues by Time Frame<b>', yaxis=dict(title=' Departures Count', range=[0,60000]),xaxis=dict(title='Time Frame <i>(00:00 - 23:59)<i>'),bargap=1)



fig.add_annotation( x='0600-0659', y=48000, text="<b>Highest Departures<b>",showarrow=False, font=dict( family="Calbiri", size=14, color="#ffffff" ), align="left", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363", ax=50, ay=-40, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#f29624", opacity=0.8)





fig.show()
#short <800, medium 800-2200 , long2200



df20['FlightType'] = 'Short Haul'

df20.loc[(df20['DISTANCE'] >= 800) & (df20['DISTANCE'] <= 2200), 'FlightType'] = 'Medium Haul'

df20.loc[(df20['DISTANCE'] > 2200), 'FlightType'] = 'Long Haul'



df_flight = pd.DataFrame(df20['FlightType'].value_counts().reset_index().values, columns=["FlightType", "AggregateType"])

labels = ["Short Haul","Medium Haul","Long Haul"]

value = [df_flight['AggregateType'][0],df_flight['AggregateType'][1],df_flight['AggregateType'][2]]

# colors=['lightcyan','cyan','royalblue']

figs = go.Figure(data=[go.Pie(labels=labels, values=value, pull=[0, 0, 0.3],textinfo = 'label+percent', hole = 0.3, hoverinfo="label+percent")])

figs.update_traces( textinfo='label + percent', textfont_size=10)

figs.update_layout(

    title_text="<b>Fligts By Distance<b> <i>(in Miles)<i>",title_x = 0.5, font_size = 12,legend=dict(x=.75, y=0.55),

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='<b>Distance<b>', x=0.5, y=0.5, font_size=11, showarrow=False)]

)

figs.show()
#Longest Air Routes

df_sub20 = df20[['ORIGIN','DEST','DISTANCE']].drop_duplicates()

df_sub20 = df_sub20.sort_values('DISTANCE',ascending=False)

df_sub20 = df_sub20[['ORIGIN','DEST']]

subList20 = df_sub20.head(20).values.tolist()

subList20 = list(set(tuple(x) for x in subList20))

g20 = nx.from_edgelist(subList20)

len(g20.nodes()), len(g20.edges())





plt.figure(figsize=(20,10))

# <matplotlib.figure.Figure object at 0x7f1b65ea5e80>



nx.draw(g20, pos = nx.nx_pydot.graphviz_layout(g20), \

    node_size=1200, node_color='lightblue', linewidths=2, \

    font_size=10, font_weight='bold', with_labels=True, dpi=10000)

plt.title("Top 20 Longest routes by Distance", fontdict=None, loc='Left', fontsize = 14)

plt.show()    ## plot2.png attached
#Smallest Air Routes

df_sub20 = df20[['ORIGIN','DEST','DISTANCE']].drop_duplicates()

df_sub20 = df_sub20.sort_values('DISTANCE',ascending=True)

df_sub20 = df_sub20[['ORIGIN','DEST']]

subList20 = df_sub20.head(20).values.tolist()

subList20 = list(set(tuple(x) for x in subList20))

g20 = nx.from_edgelist(subList20)

len(g20.nodes()), len(g20.edges())





plt.figure(figsize=(20,10))

# <matplotlib.figure.Figure object at 0x7f1b65ea5e80>



nx.draw(g20, pos = nx.nx_pydot.graphviz_layout(g20), \

    node_size=1200, node_color='lightblue', linewidths=2, \

    font_size=10, font_weight='bold', with_labels=True, dpi=10000)

plt.title("Top 20 Shortest routes by Distance", fontdict=None, loc='Left', fontsize = 14)

plt.show()    ## plot2.png attached
#All Air routes

df_sub = df20[['ORIGIN','DEST']]

subList = df_sub.values.tolist()

subList = list(set(tuple(x) for x in subList))

g = nx.from_edgelist(subList)

len(g.nodes()), len(g.edges())



plt.figure(figsize=(20,14))

# <matplotlib.figure.Figure object at 0x7f1b65ea5e80>



nx.draw(g, pos = nx.nx_pydot.graphviz_layout(g), \

    node_size=1200, node_color='lightblue', linewidths=2, \

    font_size=10, font_weight='bold', with_labels=True, dpi=10000)

plt.title("All routes by Distance", fontdict=None, loc='Left', fontsize = 14)

plt.show()    ## plot2.png attached
import missingno as msno

plt.figure(figsize=(5,5))

msno.bar(df20)
msno.heatmap(df20) 
f, ax = plt.subplots(figsize=(10, 8))

corr = df20.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)