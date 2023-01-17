import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from  pandas import json_normalize
#sns.set_style('whitegrid')
#from plotly.offline import init_notebook_mode, iplot, plot
#import plotly as py
#import plotly.graph_objs as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dat = pd.read_json('/kaggle/input/covid19-stream-data/json')
df = json_normalize(dat['records'])
df
df.dateRep = pd.to_datetime(df.dateRep, format="%d/%m/%Y")
df[df.countriesAndTerritories=='']
df.info()
df[df.countriesAndTerritories == 'United_Kingdom'].sort_values('dateRep')
df.head(10)
df = df.sort_values('dateRep')
df.groupby('continentExp').sum()
df_reg=df.groupby(['continentExp']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False).reset_index()
df_reg.head(10)
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(
    columnwidth = [50],
    header=dict(values=('continentExp', 'cases', 'deaths'),
                fill_color='#104E8B',
                align='center',
                font_size=14,
                font_color='white',
                height=40),
    cells=dict(values=[df_reg['continentExp'].head(10), df_reg['cases'].head(10), df_reg['deaths'].head(10)],
               fill=dict(color=['#509EEA', '#A4CEF8',]),
               align='right',
               font_size=12,
               height=30))
])

fig.show()
df_country=df.groupby(['dateRep','continentExp']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False)
df_country.head(10)
dfd = df_country.groupby('dateRep').sum()
dfd.head()
import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


init_notebook_mode(connected=True)

print(__version__) # requires version >= 1.9.0
cf.go_offline()
dfd[['cases','deaths']].iplot(title = 'World Situation Over Time')
df_country.head()

df_glob = df_country.groupby(['dateRep', 'continentExp'])['cases', 'deaths'].max().reset_index()
df_glob['Dates'] = pd.to_datetime(df_glob['dateRep'])
df_glob['Dates'] = df_glob['Dates'].dt.strftime('%d/%m/%Y')
df_glob['size'] = df_glob['cases'].pow(0.3)

fig = px.scatter_geo(df_glob, locations='continentExp', locationmode='country names', 
                     color="cases", hover_name="continentExp", 
                     range_color= [0, 1500], 
                     projection="mercator", animation_frame='Dates', 
                     title='COVID-19: World Spread', color_continuous_scale="portland")
fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()