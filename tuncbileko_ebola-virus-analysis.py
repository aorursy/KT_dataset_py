# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import datetime

import plotly.io as pio



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ebola = pd.read_csv("/kaggle/input/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv") 

total = pd.read_csv("/kaggle/input/ebola-total/total.csv")
print(total.head(10))
#fixing 

ebola= ebola[['Date', 'Country', 'No. of confirmed, probable and suspected cases',

                     'No. of confirmed, probable and suspected deaths']]

ebola.columns = ['Date', 'Country', 'Cases', 'Deaths']

ebola['Date'] = ebola['Date'].astype('datetime64')

ebola['year'] = pd.DatetimeIndex(ebola['Date']).year
print(ebola.info())

print(ebola.head())
#Grouping data into years

e_2014 = ebola[ebola['year'] == 2014].reset_index()

e_2014_grp = e_2014.groupby('Country')['Cases', 'Deaths'].sum().reset_index()



e_2015 = ebola[ebola['year'] == 2015].reset_index()

e_2015_grp = e_2015.groupby('Country')['Cases', 'Deaths'].sum().reset_index()





e_2016 = ebola[ebola['year'] == 2016].reset_index()

e_2016_grp = e_2016.groupby('Country')['Cases', 'Deaths'].sum().reset_index()





ebola_total = ebola

ebola_total = ebola_total.groupby(['Date', 'Country'])['Cases', 'Deaths'].sum()

ebola_total = ebola_total.reset_index()

print(ebola_total['Country'].value_counts().count())
fig = px.choropleth(total, locations="country", locationmode='country names',

                    color="cases", hover_name="country", 

                            scope ="africa",

                    color_continuous_scale="dense",

                     labels={'cases':'cases'},

                     title={

                         'text': "EBOLA Total Cases",

                         'y':0.9,

                         'x':0.5,

                         'xanchor': 'center',

                         'yanchor': 'top'})

fig.show()
fig = px.choropleth(total, locations="country", locationmode='country names',

                    color="deaths", hover_name="country", 

                            scope ="africa",

                    color_continuous_scale="dense",

                     labels={'cases':'cases'},

                     title={

                         'text': "EBOLA Total Deaths",

                         'y':0.9,

                         'x':0.5,

                         'xanchor': 'center',

                         'yanchor': 'top'})

fig.show()
fig = px.treemap(total.sort_values(by='cases', ascending=False).reset_index(drop=True), 

                 path=["country"], values="cases",

                 color_continuous_scale="dense")

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.treemap(total.sort_values(by='deaths', ascending=False).reset_index(drop=True), 

                 path=["country"], values="deaths",

                 color_continuous_scale="dense")

fig.data[0].textinfo = 'label+text+value'

fig.show()
import plotly.graph_objects as go

animals=total['country']

cases =total['cases']

deaths = total['deaths']



fig = go.Figure(data=[

    go.Bar(name='Cases', x=animals, y=cases),

    go.Bar(name='Deaths', x=animals, y=deaths),

])

# Change the bar mode

fig.update_layout(

                  title={

        'text': "Ebola",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

                  barmode='group', 

                  xaxis={'categoryorder':'total descending'})

fig.show()
temp2 = ebola.groupby(['Date','Country'])['Cases','Deaths'].sum().reset_index()

temp2.head()
fig = px.line(temp2, x="Date", y="Cases", color='Country',width=800, height=400)

fig.show()
fig = px.line(temp2, x="Date", y="Deaths", color='Country',width=800, height=400)

fig.show()
