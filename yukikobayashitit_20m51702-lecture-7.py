import pandas as pd

import plotly.express as px



df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



fig = px.choropleth(df,locations='Country/Region', \

                     locationmode='country names', \

                     color='Confirmed', \

                     hover_name='Country/Region', \

                     animation_frame='ObservationDate', \

                     color_continuous_scale='Rainbow',

                     range_color=(0.1,100000.)

                    )

fig.update_layout(title_text='Confirmed Cumulative Cases per Country', title_x=0.5)

fig.show()
fig.write_html('Confirmed_Cumulative_Cases_per_Country_20M51702.html')
df['daily_existing'] = df['Confirmed'].values - df['Deaths'].diff() - df['Recovered'].diff()



fig2 = px.choropleth(df, \

                     locations='Country/Region', \

                     locationmode='country names', \

                     color='daily_existing', \

                     hover_name='Country/Region', \

                     animation_frame='ObservationDate', \

                     color_continuous_scale='Rainbow',

                     range_color=(0.1,100000.)

                    )

fig2.update_layout(title_text='Remaining Confirmed Cases per Country of each day', title_x=0.5)

fig2.show()
fig2.write_html('Remaining_Confirmed_Cases_per_Country_ofeachday_20M51702.html')
df = pd.read_csv('../input/april-25-2015-nepal-earthquake/query.csv',header=0)



# Convert string index to datetime object

df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')



# Mapping (I selected the range from magnitude 1 to 7)

fig3 = px.scatter_geo(df, \

                     lat='latitude', \

                     lon='longitude', \

                     color='mag', \

                     animation_frame='time', \

                     color_continuous_scale='Rainbow', \

                     range_color=(1.,7.)

                    )



fig3.update_layout(title_text='Hourly global map of earthquake epicenters April 25,2015', title_x=0.5)

fig3.show()
fig3.write_html('Hourly_global_map_of_earthquake_epicenters_20M51702.html')