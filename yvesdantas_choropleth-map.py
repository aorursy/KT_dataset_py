import numpy as np
import pandas as pd
import geopandas as gpd
import folium 
import os
import json
import warnings 
warnings.filterwarnings('ignore')
state_geo = gpd.read_file(os.path.join('../input/States_IBGE.json'))
state_population = os.path.join('../input/populacao-e-pib-por-estados.xlsx')
state_data = pd.read_excel(state_population)
state_data.head()
m = folium.Map(location=[-15.7, -47.8], zoom_start=4)

m.choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['Estado', 'População'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='População',
)

folium.LayerControl().add_to(m)

m