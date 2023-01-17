import pandas as pd

import plotly.express as px



cf = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

cf = cf.groupby(['ObservationDate', 'Country/Region']).sum().reset_index()



cf['Daily existing'] = cf['Confirmed'].values - cf['Deaths'].diff() - cf['Recovered'].diff()

fig = px.choropleth(cf, locations='Country/Region', locationmode='country names', color='Daily existing', hover_name='Country/Region', animation_frame='ObservationDate', color_continuous_scale='Rainbow')

fig.update_layout(title_text='Confirmed Cumulative cases per country', title_x=0.5)

fig.show()
df = pd.read_csv('../input/earthquake-200614/query.csv', header=0)



df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='mag', animation_frame='time', color_continuous_scale='Rainbow')

fig.show()