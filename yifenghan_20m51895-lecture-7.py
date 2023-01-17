import pandas as pd

import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header = 0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig=px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Sunsetdark',range_color=(0.1,150000.))

fig.update_layout(title_text = 'Confirmed Cumulative Cases per Country',title_x = 0.5)

fig.show()
df['Existing cases'] = df['Confirmed']-df['Deaths']-df['Recovered']

fig2=px.choropleth(df,locations='Country/Region',locationmode='country names',color='Existing cases',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='PuRd',range_color=(0.1,150000.))

fig2.update_layout(title_text = 'Existing cases per Country',title_x = 0.5)

fig2.show()

fig.write_html('Existing cases per Country')
df = pd.read_csv('../input/earthquake-2008512/query.csv',header = 0)

df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig2=px.scatter_geo(df,lat = 'latitude',lon = 'longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(2,9.))

fig2.update_layout(title_text = 'Earthquake happened in 2008-5-12',title_x = 0.5)

fig2.show()