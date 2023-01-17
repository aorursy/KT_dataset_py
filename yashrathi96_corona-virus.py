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
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

#time_series_2019_ncov_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

#time_series_2019_ncov_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

#time_series_2019_ncov_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
data.describe().T
data.columns
data.Country.unique()
data.Country = data.Country.replace('Mainland China','China')
data = data.drop(['Sno','Date'], axis=1)
data[['Province/State','Country']] = data[['Province/State','Country']].fillna('Unknown')

data[['Confirmed', 'Deaths', 'Recovered']] = data[['Confirmed', 'Deaths', 'Recovered']].fillna(0.0)
latest_data = data.groupby('Country')['Last Update','Confirmed','Deaths','Recovered'].max().reset_index()

latest_data
date_data = data.groupby(['Country', 'Last Update']).sum().reset_index().sort_values('Last Update', ascending=False)

date_data = date_data.groupby(['Country']).max().reset_index().sort_values('Last Update')

date_data["Size"] = np.where(date_data['Country']=='China', date_data['Confirmed'], date_data['Confirmed']*300)
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(date_data, locations="Country", locationmode = "country names",

                     hover_name="Country", size="Size", color="Confirmed",

                     animation_frame="Last Update", 

                     projection="natural earth",

                     title="Progression of Coronavirus in Confirmed Cases",template="none")

fig.show()
max_death=latest_data.sort_values(by = 'Deaths',ascending=False).reset_index()

max_death=max_death.head(5)

max_death

no_china_data = latest_data.groupby(['Country'])['Confirmed', 'Recovered', 'Deaths'].sum().reset_index()

no_china_data = no_china_data[(no_china_data['Country'] != 'China') & (no_china_data['Country'] != 'Others')]
import plotly.graph_objects as go



fig = go.Figure(go.Bar(x=no_china_data['Confirmed'], y=no_china_data['Country'], name='Confirmed', orientation = 'h'))

fig.add_trace(go.Bar(x=no_china_data['Deaths'], y=no_china_data['Country'], name='Deaths', orientation = 'h'))

fig.add_trace(go.Bar(x=no_china_data['Recovered'], y=no_china_data['Country'], name='Recovered', orientation = 'h'))



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, height = 1000)

fig.show()
only_china_data = data[data['Country']=='China']

only_china_data
only_china_data = only_china_data.groupby(['Province/State'])['Confirmed','Deaths','Recovered'].sum().reset_index()

only_china_data
import plotly.graph_objects as go



fig = go.Figure(go.Bar(x=only_china_data['Confirmed'], y=only_china_data['Province/State'], name='Confirmed', orientation = 'h'))

fig.add_trace(go.Bar(x=only_china_data['Deaths'], y=only_china_data['Province/State'], name='Deaths', orientation = 'h'))

fig.add_trace(go.Bar(x=only_china_data['Recovered'], y=only_china_data['Province/State'], name='Recovered', orientation = 'h'))



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, height = 1000)

fig.show()
hubei_data = only_china_data[only_china_data['Province/State']=='Hubei']

hubei_data = hubei_data.drop(['Province/State'], axis=1)

hubei_data
import plotly.graph_objects as go

labels = ['Confirmed','Deaths','Recovered']

values =[361095,10690,19965]

fig=go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    title_text="Hubei, China")

fig.show()
hubei_data = data[data['Province/State']=='Hubei']
import plotly.express as px



df = px.data.gapminder().query("country=='Canada'")

fig = px.line(hubei_data, x="Last Update", y="Confirmed", title='Confirmed Cases in Hubei')

fig.show()