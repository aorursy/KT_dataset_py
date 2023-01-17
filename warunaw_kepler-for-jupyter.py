import numpy as np

import pandas as pd



from keplergl import KeplerGl

import geopandas as gpd



import os

print(os.listdir("../input"))
#Create a basemap 

map = KeplerGl(height=500, width=800)



map
df = gpd.read_file("../input/nypdcomplaints/NYPD_Complaint_Data_Year_To_Date_V2.csv")
df.columns

df.isnull().sum()
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')

df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

df.dtypes
# Create a geopdataframe

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
# Add data to Kepler

map.add_data(data=gdf, name='crimes')
neighborhoods = gpd.read_file('../input/nycneighborhood/NYC_NEIGHBORHOODS.geojson')
# neighborhoods
# from geopandas.tools import sjoin

# sjoined_listings = gpd.sjoin(gdf, neighborhoods, op='within')
buildings = gpd.read_file("../input/nyhousing/Housing_New_York_Units_by_Building.csv")

buildings['Latitude'] = pd.to_numeric(buildings['Latitude'], errors='coerce')

buildings['Longitude'] = pd.to_numeric(buildings['Longitude'], errors='coerce')
gdf_buldings = gpd.GeoDataFrame(buildings, geometry=gpd.points_from_xy(buildings.Longitude, buildings.Latitude))
map2