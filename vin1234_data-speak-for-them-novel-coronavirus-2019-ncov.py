import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

%matplotlib inline



import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)



from datetime import date, datetime,timedelta



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

import os
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line
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
## The below dataset is the modified version of the originally uploaded initial data

df=pd.read_csv('/kaggle/input/final_corona_virus_data.csv')
df.head()
import json

with open('/kaggle/input/Geocoding_api_data.json') as f:

    d = json.load(f)

##Looking at data types

df.info()
# plot the world data for 22nd January.



df_22Jan=df[df['Date']=='2020-01-22']
df_22Jan
dates = np.sort(df_22Jan['Date'].unique())

data = [go.Scattergeo(

            locationmode='country names',

            lon = df_22Jan['lon'],lat = df_22Jan['lat'],

            text = df_22Jan['Country'] + ', ' + df_22Jan['PS'] +   '-> Deaths: ' + df_22Jan['Deaths'].astype(str) + ' Confirmed: ' + df_22Jan['Confirmed'].astype(str),

            mode = 'lines+markers',

            marker=dict(size=(df_22Jan['Confirmed'])**(1/1.9)+9,color='red')



) ]
fig = go.Figure(

    data=data[0],

    layout=go.Layout(

        title = {'text': f'[Day -1]Detection of Corona Virus 22nd January','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'},

        geo = dict(projection_type='robinson',showland = True,landcolor = "rgb(252, 240, 220)",showcountries=True,showocean=True,countrycolor = "rgb(128, 128, 128)",

            )),

    

    frames=[go.Frame(data=dt, layout=go.Layout(title={'text': f'Corona Virus','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'}))for dt,date in zip(data[1:],dates[1:])])



# fig.show()

py.offline.iplot(fig)
# plot the world data for 22nd January.



df_2ndFeb=df[df['Date']=='2020-02-02']
df_2ndFeb.head()
dates = np.sort(df_2ndFeb['Date'].unique())

data = [go.Scattergeo(

            locationmode='country names',

            lon = df_2ndFeb['lon'],lat = df_2ndFeb['lat'],

            text = df_2ndFeb['Country'] + ', ' + df_2ndFeb['PS'] +   '-> Deaths: ' + df_2ndFeb['Deaths'].astype(str) + ' Confirmed: ' + df_2ndFeb['Confirmed'].astype(str),

            mode = 'lines+markers',

#             marker = dict(size = (df_22Jan['Confirmed'])**(1/2.7)+9,opacity = 0.6,reversescale = True,

#                 autocolorscale = False,line = dict(width=0.5,color='rgba(250, 0, 250)'),cmin=0,color=df_22Jan['Deaths'],

#                 cmax=df_22Jan['Deaths'].max(),

#                 colorbar_title="Number of Deaths"

    

                          

#             )

            marker=dict(size=(df_2ndFeb['Confirmed'])**(1/2.9)+9,color='red')



) ]
fig = go.Figure(

    data=data[0],

    layout=go.Layout(

        title = {'text': f'[Last recorded data]Detection of Corona Virus 2nd Feb','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'},

        geo = dict(projection_type='robinson',showland = True,landcolor = "rgb(252, 240, 220)",showcountries=True,showocean=True,countrycolor = "rgb(128, 128, 128)",

            )),

    

    frames=[go.Frame(data=dt, layout=go.Layout(title={'text': f'Corona Virus','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'}))for dt,date in zip(data[1:],dates[1:])])



fig.show()
dates = np.sort(df['Date'].unique())

data = [go.Scattergeo(

            locationmode='country names',

            lon = df.loc[df['Date']==dt, 'lon'],

            lat = df.loc[df['Date']==dt, 'lat'],

            text = df.loc[df['Date']==dt, 'Country'] + ', ' + df.loc[df['Date']==dt, 'PS'] +   '-> Deaths: ' + df.loc[df['Date']==dt, 'Deaths'].astype(str) + ' Confirmed: ' + df.loc[df['Date']==dt,'Confirmed'].astype(str),

            mode = 'lines+markers',

            marker = dict(

                size = (df.loc[df['Date']==dt,'Confirmed'])**(1/2.7)+3,

                reversescale = True,

                autocolorscale = False,

                line = dict(

                    width=0.5,

                    color='rgba(0, 0, 0)'

                        ),

                #colorscale='rdgy', #'jet',rdylbu, 'oryel', 

                cmin=0,

                color=df.loc[df['Date']==dt,'Deaths'],

                cmax=df['Deaths'].max(),

                colorbar_title="Number of Deaths"

            )) 

        for dt in dates]





fig = go.Figure(

    data=data[0],

    layout=go.Layout(

        title = {'text': f'Corona Virus, {dates[0]}',

                                'y':0.98,

                                'x':0.5,

                                'xanchor': 'center',

                                'yanchor': 'top'},

        geo = dict(

            scope='world',

            projection_type='robinson',

            showland = True,

            landcolor = "rgb(252, 240, 220)",

            showcountries=True,

            showocean=True,

            countrycolor = "rgb(128, 128, 128)"

            

            ),

     updatemenus=[dict(

            type="buttons",

            buttons=[dict(label="Play",

                          method="animate",

                          args=[None])])]),

    

    frames=[go.Frame(data=dt, 

                     layout=go.Layout(

                          title={'text': f'Corona Virus, {date}',

                                'y':0.98,

                                'x':0.5,

                                'xanchor': 'center',

                                'yanchor': 'top'}

                           ))

            for dt,date in zip(data[1:],dates[1:])])



fig.show()

# For China

china=df[df['Country']=='China']

china.head()
# For remaining world

world=df[df['Country']!='China']

world.tail()
W=world.groupby('Date')['Confirmed'].sum()
C=china.groupby('Date')['Confirmed'].sum()
import plotly.graph_objects as go

import datetime



X_china = C.index

X_world = W.index



fig = go.Figure(layout=go.Layout(

        title = {'text': f'Corona Virus Confirmed cases in China vs Rest World',

                                'y':0.85,

                                'x':0.5,

                                'xanchor': 'center',

                                'yanchor': 'top'},

        xaxis_title="Virus Spread Dates",

        yaxis_title="Number of Confirmed infected people"))



fig.add_trace(go.Scatter(x=X_world, y=W.values,line_color='deepskyblue',name='Rest_World',

                opacity=0.8))



fig.add_trace(go.Scatter(x=X_china, y=C.values,line_color='red',name='China',

                opacity=0.6))

fig.show()
# For China

death_c=china.groupby('Date')['Deaths'].sum()



# Rest world

death_w=world.groupby('Date')['Deaths'].sum()
X_c = death_c.index

X_w = death_w.index



fig = go.Figure(layout=go.Layout(

        title = {'text': f'Number of deaths due to Corona Virus China vs Rest World',

                                'y':0.85,

                                'x':0.5,

                                'xanchor': 'center',

                                'yanchor': 'top'},

        xaxis_title="Virus Spread Dates",

        yaxis_title="Number of Deaths"))



fig.add_trace(go.Scatter(x=X_w, y=death_w.values,line_color='deepskyblue',name='Rest_World',

                opacity=0.8))



fig.add_trace(go.Scatter(x=X_c, y=death_c.values,line_color='red',name='China',

                opacity=0.6))

fig.show()