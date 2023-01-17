import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df.fips[df["county"]== 'New York City'] = 36061
#df['fips']= df['fips'].astype(int)
#df['fips'] = df['fips'].astype(str)
pop = pd.read_excel('../input/pop-us-2018/Pop_2018.xlsx')
df_pop = df.join(pop.set_index('FIPS*'), on='fips')
df_pop = df_pop.drop(["County name", "State"],axis=1)
df_filter_n = df_pop[df_pop["state"]=="New York"]
px.line(df_filter_n, x="date", y="cases", color ="county")
df_filter_c = df_pop[df_pop["state"]=="California"]
px.line(df_filter_c, x="date", y="cases", color ="county")
df_filter_i = df_pop[df_pop["state"]=="Illinois"]
px.line(df_filter_i, x="date", y="cases", color ="county")
import numpy as np
df_filter_d = df_pop[df_pop["date"]==df.date.max()]
df_filter_d.fips[np.isnan(df_filter_d["Pop.2018"]) == True ] = 111111
df_filter_d['fips'] = df_filter_d['fips'].astype(int)
df_filter_d['fips'] = df_filter_d['fips'].astype(str)
df_filter_d['fips'] = df_filter_d['fips'].str.zfill(5) 
df_county_map =  df_filter_d.groupby(['fips'])['cases'].sum()
df_county_map = df_county_map.to_frame()
df_county_map = df_county_map.reset_index()
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
fig = px.choropleth_mapbox(df_county_map, geojson=counties, locations='fips', color='cases',
                           color_continuous_scale="Viridis",range_color=(0, 50),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'cases':'confirmed cases'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig = px.choropleth_mapbox(df_filter_d, geojson=counties, locations='fips', color='Pop.2018',
                           color_continuous_scale="Viridis",range_color=(0, 500000),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'cases':'confirmed cases'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
import plotly.graph_objects as go


fig = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df_county_map.fips, z=df_county_map.cases,
                                    colorscale="Viridis", zmin=0, zmax=100,
                                    marker_opacity=0.5, marker_line_width=0))
fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


df_county_20 = df_filter_d.sort_values(by=['cases'],ascending=False).head(20)
df_county_20 = df_county_20.reset_index()

df_county_20 = df_county_20.drop(['index', 'date', 'county', 'state', 'cases', 'deaths', 'Pop.2018'], axis=1)
df_county_20.columns = ["fips1"]
df_county_20['fips1'] = df_county_20['fips1'].astype(int)
df_high_county = df_county_20.join(df_pop.set_index('fips'), on='fips1')
df_high_county = df_high_county.dropna(how='any',axis=0)
df_high_county["county_state"] = df_high_county["county"] +"," + df_high_county["state"]
px.line(df_high_county, x="date", y="cases", color ="county_state")


