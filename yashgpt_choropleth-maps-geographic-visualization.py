import plotly.graph_objs as go 

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)
import pandas as pd
df=pd.read_csv('../input/2014_World_Power_Consumption.csv')
df.head()
data = dict(

        type = 'choropleth',

        colorscale = 'Viridis',

        reversescale = True,

        locations = df['Country'],

        locationmode = "country names",

        z = df['Power Consumption KWH'],

        text = df['Country'],

        colorbar = {'title' : 'Power Consumption KWH'},

      ) 



layout = dict(title = '2014 Power Consumption KWH',

                geo = dict(showframe = False,projection = {'type':'stereographic'})

             )
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
df=pd.read_csv('../input/2012_Election_Data.csv')
df.head()
data = dict(

        type = 'choropleth',

        colorscale = 'Viridis',

        reversescale = True,

        locations = df['State Abv'],

        locationmode = "USA-states",

        z = df['Voting-Age Population (VAP)'],

        text = df['State'],

        marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

        colorbar = {'title' : 'Voting-Age Population (VAP)'},

      ) 
layout = dict(title = '2012_Election_Data',

                geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)