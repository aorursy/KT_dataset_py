import pandas as pd

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['Remaining'] = df['Confirmed']-df['Deaths']-df['Recovered']
fig = px.choropleth(df, locations='Country/Region', locationmode='country names', hover_name='Country/Region', color='Remaining', animation_frame='ObservationDate', color_continuous_scale='YlOrRd', range_color=(0.,100000.))

fig.update_layout(title='Worldwide Remaining Infection Cases')

fig.show()

fig.write_html('Wiranpat.html')
import datetime

df = pd.read_csv('../input/earthquake2/earthquake2.csv', header=0)
df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='mag', color_continuous_scale='Rainbow', range_color=(0.,7.), animation_frame='time')

fig.update_layout(title='Hourly epicenter of earthquake on April 25-26, 2015')

fig.show()

fig.write_html('Wiranpat-earthquake.html')