# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

full_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

full_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
train.tail()
full_data.head()
full_deaths.head()
import plotly.graph_objects as go



full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_data['text'] = full_data['Country'] + " " + full_data["4/13/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_data["Long"],

    lat = full_data["Lat"],

    text = full_data["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_data['4/13/20'],

        cmax = full_data['4/13/20'].max(),

        colorbar_title = "COVID 19 Reported Cases"

    )

))



fig.update_layout(

    title = "COVID19 Confirmed Cases Around the World on 13th April",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_data['text'] = full_data['Country'] + " " + full_data["4/30/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_data["Long"],

    lat = full_data["Lat"],

    text = full_data["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_data['4/30/20'],

        cmax = full_data['4/30/20'].max(),

        colorbar_title = "COVID 19 Reported Cases"

    )

))



fig.update_layout(

    title = "COVID19 Confirmed Cases Around the World on 30th April",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_data['text'] = full_data['Country'] + " " + full_data["5/4/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_data["Long"],

    lat = full_data["Lat"],

    text = full_data["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_data['5/4/20'],

        cmax = full_data['5/4/20'].max(),

        colorbar_title = "COVID 19 Reported Cases"

    )

))



fig.update_layout(

    title = "COVID19 Confirmed Cases Around the World on 4th May",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["4/13/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_deaths["Long"],

    lat = full_deaths["Lat"],

    text = full_deaths["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_deaths['4/13/20'],

        cmax = full_deaths['4/13/20'].max(),

        colorbar_title = "COVID 19 Deaths"

    )

))



fig.update_layout(

    title = "COVID19 Deaths Around the World on 13th April",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["4/30/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_deaths["Long"],

    lat = full_deaths["Lat"],

    text = full_deaths["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_deaths['4/30/20'],

        cmax = full_deaths['4/30/20'].max(),

        colorbar_title = "COVID 19 Deaths"

    )

))



fig.update_layout(

    title = "COVID19 Deaths Around the World on 30th April",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})



full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["5/4/20"].astype(str)

fig = go.Figure(data = go.Scattergeo(

    lon = full_deaths["Long"],

    lat = full_deaths["Lat"],

    text = full_deaths["text"],

    mode = "markers",

    marker = dict(

        size = 12,

        opacity = 0.8,

        reversescale = True,

        autocolorscale = True,

        symbol = 'square',

        line = dict(

            width = 1,

            color = 'rgba(102, 102, 102)'

        ),

        cmin = 0,

        color = full_deaths['5/4/20'],

        cmax = full_deaths['5/4/20'].max(),

        colorbar_title = "COVID 19 Deaths"

    )

))



fig.update_layout(

    title = "COVID19 Deaths Around the World on 4th May",

    geo = dict(

        scope = "world",

        showland = True,

    )

)





fig.show()
US = full_data[full_data['Country'] == 'US']

India = full_data[full_data['Country'] == 'India']

Italy = full_data[full_data['Country'] == 'Italy']

SK = full_data[full_data['Country'] == 'Korea, South']

US
US.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)

columns = US.columns



US_data = pd.DataFrame(columns=['day', 'cases'])

index = 1

for col in columns:

    US_data.loc[len(US_data)] = [index, US[col].values[0]]

    index += 1



# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------

India.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)

columns = India.columns



India_data = pd.DataFrame(columns=['day', 'cases'])

index = 1

for col in columns:

    India_data.loc[len(India_data)] = [index, India[col].values[0]]

    index += 1

    

# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------

Italy.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)

columns = Italy.columns



Italy_data = pd.DataFrame(columns=['day', 'cases'])

index = 1

for col in columns:

    Italy_data.loc[len(Italy_data)] = [index, Italy[col].values[0]]

    index += 1

    

# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------

SK.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)

columns = SK.columns



SK_data = pd.DataFrame(columns=['day', 'cases'])

index = 1

for col in columns:

    SK_data.loc[len(SK_data)] = [index, SK[col].values[0]]

    index += 1
plt.scatter(US_data['day'], US_data['cases'])
plt.scatter(India_data['day'], India_data['cases'])
plt.scatter(Italy_data['day'], Italy_data['cases'])
plt.scatter(SK_data['day'], SK_data['cases'])