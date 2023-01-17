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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#load the data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

codes = pd.read_csv('/kaggle/input/country-codes/country_stats.csv')
train['Date'] = pd.to_datetime(train['Date'])

train['Date'] = train['Date'].dt.strftime('%m/%d/%Y')
col = ['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',

       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20']
df['Province/State'].fillna('Others', inplace=True)
#melting the data for getting the recovered counts

temp=df.melt(id_vars=('Province/State', 'Country/Region'),value_vars=['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',

       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20'],value_name='Recovered')
temp['variable'] = pd.to_datetime(temp['variable'])

temp['variable'] = temp['variable'].dt.strftime('%m/%d/%Y')
train = pd.merge(train, temp, left_on=['Province/State','Country/Region','Date'], right_on=['Province/State','Country/Region','variable'], how='left')
train = train.sort_values(by=['Country/Region','Province/State','Date'])
train.fillna(0, inplace=True)
train['Active'] = train['ConfirmedCases'] - train['Fatalities'] - train['Recovered']
train['Fatalities_perc'] = 100*(train['Fatalities']/train['ConfirmedCases'])

train['Recovered_Perc'] = 100*(train['Recovered']/train['ConfirmedCases'])

train['Active_perc'] = 100*(train['Active']/train['ConfirmedCases'])
train['Active'] = train['Active'].clip(lower=0)
train.drop(columns=['variable'], inplace=True)
train.isnull().sum()
train.fillna(0, inplace=True)

train.head(2)
del df

del temp
import gc

gc.collect()
codes.head(2)
train = pd.merge(train, codes[['country_name','country_code3']],left_on=['Country/Region'], right_on=['country_name'], how='left')
train.drop(columns=['country_name'], inplace=True)
train.head()
temp=train.groupby(['Country/Region'])['ConfirmedCases'].sum().reset_index().sort_values(by='ConfirmedCases').tail(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='ConfirmedCases')

fig.show()
temp=train.groupby(['Country/Region'])['Fatalities'].sum().reset_index().sort_values(by='Fatalities').tail(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='Fatalities')

fig.show()
temp=train.groupby(['Country/Region'])['Active'].sum().reset_index().sort_values(by='Active').tail(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='Active')

fig.show()
import plotly.express as px

fig = px.scatter_mapbox(train, lat="Lat", lon="Long", hover_name="Country/Region", hover_data=["Country/Region"],

                        color_discrete_sequence=["fuchsia"], zoom=0.01, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
temp=train.groupby(['Date'])['ConfirmedCases','Fatalities','Recovered','Active'].sum().reset_index()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp.Date, y=temp['ConfirmedCases'], name="ConfirmedCases", line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Fatalities'], name="Fatalities",line_color='dimgray'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Recovered'], name="Recovered", line_color='red'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Active'], name="Active",line_color='yellow'))

fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

temp=train[train['Fatalities']>0]

freq = temp['Country/Region'].value_counts().reset_index().rename(columns={"index": "x"})



# Initialize figure with subplots

fig = make_subplots(

    rows=2, cols=2,

    column_widths=[0.6, 0.4],

    row_heights=[0.4, 0.6],

    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar"}],

           [            None                    , {"type": "surface"}]])



# Add scattergeo globe map of volcano locations

fig.add_trace(

    go.Scattergeo(lat=train["Lat"],

                  lon=train["Long"],

                  mode="markers",

                  hoverinfo="text",

                  showlegend=False,

                  marker=dict(color="crimson", size=4, opacity=0.8)),

    row=1, col=1

)





# Add locations bar chart

fig.add_trace(

    go.Bar(x=freq["x"][0:10],y=freq["Country/Region"][0:10], marker=dict(color="crimson"), showlegend=False),

    row=1, col=2

)





# Update geo subplot properties

fig.update_geos(

    projection_type="orthographic",

    landcolor="white",

    oceancolor="MidnightBlue",

    showocean=True,

    lakecolor="LightBlue"

)



# Rotate x-axis labels

fig.update_xaxes(tickangle=45)



# Set theme, margin, and annotation in layout

fig.update_layout(

    template="plotly_dark",

    margin=dict(r=10, t=25, b=40, l=60),

    annotations=[

        dict(

            

            showarrow=False,

            xref="paper",

            yref="paper",

            x=0,

            y=0)

    ]

)



fig.show()
import plotly.express as px

fig = px.density_mapbox(train, lat='Lat', lon='Long', z='Fatalities', radius=10,

                        center=dict(lat=0, lon=180), zoom=0,

                        mapbox_style="stamen-terrain")

fig.show()
import folium 

world_map_recovered = folium.Map(location=[30, 0], zoom_start=1.5,tiles='Stamen Toner')

world_data_totaly_recovered=train.copy()#world_data[world_data['confirmed']==world_data['recovered']]

for lat, lon, value, name in zip(world_data_totaly_recovered['Lat'], 

                                 world_data_totaly_recovered['Long'], 

                                 world_data_totaly_recovered['Fatalities'], 

                                 world_data_totaly_recovered['Country/Region']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Recovered</strong>: ' + str(value) + '<br>'),

                        color='green',

                        

                        fill_color='green',

                        fill_opacity=0.7 ).add_to(world_map_recovered)
world_map_recovered
temp = train.groupby(['Country/Region','country_code3'])['Fatalities','ConfirmedCases'].sum().reset_index()

temp['size'] = temp['Fatalities'].pow(0.3)

temp.head(2)
import plotly.express as px

fig = px.choropleth(temp, locations="country_code3",

                    color="size", # lifeExp is a column of gapminder

                    hover_name="Country/Region", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
train.groupby(['Country/Region'])['Fatalities'].sum().reset_index().sort_values(by='Fatalities', ascending=False).head(10).style.background_gradient(cmap='Reds')
train.groupby(['Province/State'])['ConfirmedCases','Fatalities','Recovered','Active'].sum().reset_index().sort_values(by='ConfirmedCases', ascending=False).head(20).style.background_gradient(cmap='Reds')
grouped_us= train[train['Country/Region'] == "US"].reset_index()
grouped_us.groupby(['Province/State'])['Active'].sum().reset_index().sort_values(by='Active', ascending=False).head(10).style.background_gradient(cmap='Greens')
train_China = train[train['Country/Region']=='China']
train_China.groupby(['Province/State'])['Fatalities'].sum().reset_index().sort_values(by='Fatalities', ascending=False).head(10).style.background_gradient(cmap='Reds')
import datetime as dt

temp3 = train.groupby(['Date','Country/Region'])['Fatalities','ConfirmedCases'].max().reset_index()

temp3['size'] = temp3['Fatalities'].pow(0.6)

temp3.head(10)
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="Fatalities",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 1000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="ConfirmedCases",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 2000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

temp4 = train.groupby(['Date','Country/Region'])['Fatalities','Active'].mean().reset_index()

fig = px.line(temp4, x="Date", y="Fatalities", title='Life expectancy in Canada')

fig.show()
import plotly.graph_objects as go

import numpy as np

np.random.seed(1)

# Create traces

fig = go.Figure()

#fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Fatalities'],mode='lines',name='lines'))

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Active'],mode='lines+markers',name='lines+markers'))

#fig.add_trace(go.Scatter(x=random_x, y=random_y2,mode='markers', name='markers'))

fig.show()
import plotly.express as px

fig = px.scatter(x=train['ConfirmedCases'], y=train['Active'])

fig.show()
import plotly.express as px

fig = px.scatter(x=train['ConfirmedCases'], y=train['Fatalities'])

fig.show()