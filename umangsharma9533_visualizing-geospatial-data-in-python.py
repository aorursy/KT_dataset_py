# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Point

import folium

# Import package for pretty printing

import pprint

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the dataset

permit_data = pd.read_csv('/kaggle/input/geospatical/building_permits_2017.csv')



# Look at the first few rows of the chickens DataFrame

print(permit_data.head())



# Plot the locations of all Nashville chicken permits

plt.scatter(x = permit_data.lng, y = permit_data.lat)



# Show the plot

plt.show()
service_district = gpd.read_file('/kaggle/input/geospatical/council_districts.geojson')
service_district.plot()

plt.show()
# Plot the Service Districts without any additional arguments

service_district.plot()

plt.show()



# Plot the Service Districts, color them according to name, and show a legend

service_district.plot(column = 'first_name', legend = True)

plt.show()
school_districts = gpd.read_file('/kaggle/input/geospatical/school_districts.geojson')
lgnd_kwds = {'title': 'School Districts',

               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}



# Plot the school districts using the tab20 colormap (qualitative)

school_districts.plot(column = 'district', cmap = 'tab20', legend = True, legend_kwds = lgnd_kwds)

plt.xlabel('Latitude')

plt.ylabel('Longitude')

plt.title('Nashville School Districts')

plt.show();
# Set legend style

lgnd_kwds = {'title': 'School Districts',

               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}



# Plot the school districts using the summer colormap (sequential)

school_districts.plot(column = 'district', cmap = 'summer', legend = True, legend_kwds = lgnd_kwds)

plt.xlabel('Latitude')

plt.ylabel('Longitude')

plt.title('Nashville School Districts')

plt.show();
lgnd_kwds = {'title': 'School Districts',

               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 1}



# Plot the school districts using Set3 colormap without the column argument

school_districts.plot(cmap = 'Set3', legend = True, legend_kwds = lgnd_kwds)

plt.xlabel('Latitude')

plt.ylabel('Longitude')

plt.title('Nashville School Districts')

plt.show();
# Read in the neighborhoods geojson file

neighborhoods = gpd.read_file('/kaggle/input/geospatical/neighborhoods.geojson')



# Print the first few rows of neighborhoods

print(neighborhoods.head())



# Plot the neighborhoods, color according to name and use the Dark2 colormap

neighborhoods.plot(column = 'name', cmap = 'Dark2')



# Show the plot.

plt.show()
# Print the first row of school districts GeoDataFrame and the crs

print(school_districts.head(1))

print(school_districts.crs)



# Convert the crs to epsg:3857

school_districts.geometry = school_districts.geometry.to_crs(epsg = 3857)

                        

# Print the first row of school districts GeoDataFrame and the crs again

print(school_districts.head(1))

print(school_districts.crs)
art=pd.read_csv('/kaggle/input/geospatical/public_art.csv')
# Print the first few rows of the art DataFrame

print(art.head())



# Create a geometry column from lng & lat

art['geometry'] = art.apply(lambda x: Point(float(x.Longitude), float(x.Latitude)), axis=1)



# Create a GeoDataFrame from art and verify the type

art_geo = gpd.GeoDataFrame(art, crs = neighborhoods.crs, geometry = art.geometry)

print(type(art_geo))
# Spatially join art_geo and neighborhoods 

art_intersect_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'intersects')



# Print the shape property of art_intersect_neighborhoods

print(art_intersect_neighborhoods.shape)
# Create art_within_neighborhoods by spatially joining art_geo and neighborhoods

art_within_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'within')



# Print the shape property of art_within_neighborhoods

print(art_within_neighborhoods.shape)
# Create art_within_neighborhoods by spatially joining art_geo and neighborhoods

art_containing_neighborhoods = gpd.sjoin(art_geo, neighborhoods, op = 'contains')



# Print the shape property of art_within_neighborhoods

print(art_containing_neighborhoods.shape)
# Spatially join neighborhoods with art_geo

neighborhood_art = gpd.sjoin(art_geo, neighborhoods, op = "within")



# Print the first few rows

print(neighborhood_art.head())
# Get name and title from neighborhood_art and group by name

neighborhood_art_grouped = neighborhood_art[['name', 'Title']].groupby('name')



# Aggregate the grouped data and count the artworks within each polygon

print(neighborhood_art_grouped.agg('count').sort_values(by = 'Title', ascending = False))
# Create urban_art from neighborhood_art where the neighborhood name is Urban Residents

urban_art = neighborhood_art.loc[neighborhood_art.name == "Urban Residents"]



# Get just the Urban Residents neighborhood polygon and save it as urban_polygon

urban_polygon = neighborhoods.loc[neighborhoods.name == "Urban Residents"]



# Plot the urban_polygon as ax 

ax = urban_polygon.plot(color = 'lightgreen')



# Add a plot of the urban_art and show it

urban_art.plot( ax = ax, column = 'Type', legend = True);

plt.show()
# Create downtown_center from urban_poly_3857

downtown_center = urban_polygon.geometry.centroid



# Print the type of downtown_center 

print(type(downtown_center))



# Plot the urban_poly_3857 as ax and add the center point

ax = urban_polygon.plot(color = 'lightgreen')

downtown_center.plot(ax = ax, color = 'black')

plt.xticks(rotation = 45)



# Show the plot

plt.show()
downtown_center.geometry
 #Create art_dist_meters using art and the geometry from art

art_dist_meters = gpd.GeoDataFrame(art, geometry = art.geometry, crs = {'init': 'epsg:4326'})

print(art_dist_meters.head(2))



# Set the crs of art_dist_meters to use EPSG:3857

art_dist_meters.geometry = art_dist_meters.geometry.to_crs(epsg = 3857)

print(art_dist_meters.head(2))



# Add a column to art_meters, center

art_dist_meters['center'] = art_dist_meters.apply(lambda x: downtown_center.geometry,axis=1)


art_dist_meters['center']




# Create array for folium called urban_location

urban_location = [36.16127820928791, -86.77756457127047]



# Print urban_location

print(urban_location)
# Construct a folium map with urban_location

downtown_map = folium.Map(location = urban_location, zoom_start = 15)



# Display the map

display(downtown_map)
# Construct a map from folium_loc: downtown_map

downtown_map = folium.Map(location = urban_location, zoom_start = 15)



# Draw our neighborhood: Urban Residents

folium.GeoJson(urban_polygon.geometry).add_to(downtown_map)



# Display the map

display(downtown_map)
# Iterate through the urban_art and print each part of tuple returned

for row in urban_art.iterrows():

  print('first part: ', row[0])

  print('second part: ', row[1])



# Create a location and marker with each iteration for the downtown_map

for row in urban_art.iterrows():

    row_values = row[1] 

    location = [row_values['Latitude'], row_values['Longitude']]

    marker = folium.Marker(location = location)

    marker.add_to(downtown_map)



# Display the map

display(downtown_map)
# Print the urban_art titles

print(urban_art.

    Title)



#Print the urban_art descriptions

print(urban_art.Description)



# Replace Nan and ' values in description

urban_art.Description.fillna('', inplace = True)

urban_art.Description = urban_art.Description.str.replace("'", "`")



#Print the urban_art descriptions again

print(urban_art.Description)
# Construct downtown map



folium.GeoJson(urban_polygon).add_to(downtown_map)



# Create popups inside the loop you built to create the markers

for row in urban_art.iterrows():

    row_values = row[1] 

    location = [row_values['Latitude'], row_values['Longitude']]

    popup = (str(row_values['Title']) + ': ' + 

             str(row_values['Description'])).replace("'", "`")

    marker = folium.Marker(location = location, popup = popup)

    marker.add_to(downtown_map)



# Display the map.

display(downtown_map)
# Create a shapely Point from lat and lng

permit_data['geometry'] = permit_data.apply(lambda x: Point((float(x.lng) , float(x.lat))), axis = 1)



# Build a GeoDataFrame: permits_geo

permits_geo = gpd.GeoDataFrame(permit_data, crs = service_district.crs, geometry = permit_data.geometry)



# Spatial join of permits_geo and council_districts

permits_by_district = gpd.sjoin(permits_geo, service_district, op = 'within')

print(permits_by_district.head(2))



# Create permit_counts

permit_counts = permits_by_district.groupby(['district']).size()

print(permit_counts)
# Create an area column in council_districts

service_district['area'] = service_district.geometry.area

print(service_district.head(2))
# Convert permit_counts to a DataFrame

permits_df = permit_counts.to_frame()

print(permits_df.head(2))
# Reset index and column names

permits_df.reset_index(inplace=True)

permits_df.columns = ['district', 'bldg_permits']

print(permits_df.head(2))
# Merge council_districts and permits_df: 

districts_and_permits = pd.merge(service_district, permits_df, on = 'district')

print(districts_and_permits.head(2))
# Print the type of districts_and_permits

print(type(districts_and_permits))



# Create permit_density column in districts_and_permits

districts_and_permits['permit_density'] = districts_and_permits.apply(lambda row: row.bldg_permits / row.area, axis = 1)



# Print the head of districts_and_permits

print(districts_and_permits.head(5))
# Simple plot of building permit_density

districts_and_permits.plot(column ='permit_density', legend = True);

plt.show();
# Polished choropleth of building permit_density

districts_and_permits.plot(column = 'permit_density', cmap = 'BuGn', edgecolor = 'black', legend = True)

plt.xlabel('longitude')

plt.ylabel('latitude')

plt.xticks(rotation = 'vertical')

plt.title('2017 Building Project Density by Council District')

plt.show();
# Change council_districts crs to epsg 3857

service_district = service_district.to_crs(epsg = 3857)

print(service_district.crs)

print(service_district.head())



# Create area in square km

sqm_to_sqkm = 10**6

service_district['area'] = service_district.geometry.area / sqm_to_sqkm



# Change council_districts crs back to epsg 4326

service_district = service_district.to_crs(epsg = 4326)

print(service_district.crs)

print(service_district.head())
# Create permits_geo

permits_geo = gpd.GeoDataFrame(permit_data, crs =service_district.crs, geometry = permit_data.geometry)



# Spatially join permits_geo and council_districts

permits_by_district = gpd.sjoin(permits_geo, service_district, op = 'within')

print(permits_by_district.head(2))



# Count permits in each district

permit_counts = permits_by_district.groupby('district').size()



# Convert permit_counts to a df with 2 columns: district and bldg_permits

counts_df = permit_counts.to_frame()

counts_df = counts_df.reset_index()

counts_df.columns = ['district', 'bldg_permits']

print(counts_df.head(2))
# Merge permits_by_district and counts_df

districts_and_permits = pd.merge(permits_by_district, counts_df, on = 'district')



# Create permit_density column

districts_and_permits['permit_density'] = districts_and_permits.apply(lambda row: row.bldg_permits / row.area, axis = 1)

print(districts_and_permits.head(2))



# Create choropleth plot

districts_and_permits.plot(column = 'permit_density', cmap = 'OrRd', edgecolor = 'black', legend = True)



# Add axis labels and title

plt.xlabel('longitude ')

plt.ylabel('latitude')

plt.title('2017 Building Project Density by Council District')

plt.show()
# Center point for Nashville

nashville = [36.1636,-86.7823]



# Create map

m = folium.Map(location=nashville, zoom_start=10)

# Build choropleth

m.choropleth(

    geo_data=districts_and_permits,

    name='geometry',

    data=districts_and_permits,

    columns=['district', 'permit_density'],

    key_on='feature.properties.district',

    fill_color='Reds',

    fill_opacity=0.5,

    line_opacity=1.0,

    legend_name='2017 Permitted Building Projects per km squared'

)

# Create LayerControl and add it to the map            

folium.LayerControl().add_to(m)



# Display the map

display(m)   


