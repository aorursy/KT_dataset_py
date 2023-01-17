import pandas as pd

import geopandas as gpd

import numpy as np

import folium

from folium import Circle

import matplotlib.pyplot as plt

from sklearn import preprocessing



#Function for displaying the map

#def embed_map(m, file_name):

#    from IPython.display import IFrame

#    m.save(file_name)

#    return IFrame(file_name, width='100%', height='500px')
# data needs to be transformed so we have the rows are cities.

df = pd.read_csv("/kaggle/input/cost-of-living/cost-of-living.csv", index_col=[0]).T.reset_index()

df = df.rename(columns={'index':'location'})

df.head()
!pip install opencage
from opencage.geocoder import OpenCageGeocode

from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

key = user_secrets.get_secret("key")



geocoder = OpenCageGeocode(key)



list_lat = [] 

list_long = []



for row in df.location:

    try:

        query = str(row)

        results = geocoder.geocode(query)   

        lat = results[0]['geometry']['lat']

        long = results[0]['geometry']['lng']

        list_lat.append(lat)

        list_long.append(long)

    except:

        list_lat.append(None)

        list_long.append(None)



df['lat'] = list_lat   

df['lon'] = list_long
df['city'] = df['location'].apply(lambda x: str(x).split(', ')[0])
# To find some interesting columns to plot I've sorted them by range. 

# Perhaps a better way to do this in future would be by variance.

top_range = (df.describe().loc['min',:]/df.describe().loc['max',:]).sort_values().index[2:22]

list(top_range)
def color_producer(val):

    if val <= df[item].quantile(.25):

        return 'forestgreen'

    elif val <= df[item].quantile(.50):

        return 'goldenrod'

    elif val <= df[item].quantile(.75):

        return 'darkred'

    else:

        return 'black'
m_1 = folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)



item = top_range[0]



# Add a bubble map to the base map

for i in range(0,len(df)):

    Circle(

        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],

        radius=1000,

        color=color_producer(df.iloc[i][item])).add_to(m_1)



print ('Price of: ', item)

m_1
m_2= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)

item = top_range[2]



# Add a bubble map to the base map

for i in range(0,len(df)):

    Circle(

        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],

        radius=1000,

        color=color_producer(df.iloc[i][item])).add_to(m_2)



print ('Price of: ', item)

# Display the map

#e

m_2
m_3= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)

item = top_range[9]



# Add a bubble map to the base map

for i in range(0,len(df)):

    Circle(

        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],

        radius=1000,

        color=color_producer(df.iloc[i][item])).add_to(m_3)



print ('Price of: ', item)

# Display the map

#e

m_3
m_4= folium.Map(location=[df.lat.mean(),df.lon.mean()], tiles='cartodbpositron', zoom_start=2)

item = 'Toyota Corolla 1.6l 97kW Comfort (Or Equivalent New Car)'



# Add a bubble map to the base map

for i in range(0,len(df)):

    Circle(

        location=[df.iloc[i]['lat'], df.iloc[i]['lon']],

        radius=1000,

        color=color_producer(df.iloc[i][item])).add_to(m_4)



print ('Price of: ', item)

m_4
# I spent sometime trying to set up another API using the Overpass Turbo API with OpenStreetMap for geometry data.

# Fortuantely I found Geopandas already has everything ready to import.

cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
df['country'] = df.location.apply(lambda x: str(x).split(', ')[-1])

countries = df.groupby('country', as_index=False).mean()
name_change = {'Bosnia And Herzegovina' : 'Bosnia and Herz.',

'United States' : 'United States of America',

'Czech Republic' : 'Czechia',

'Dominican Republic' : 'Dominican Rep.'}



countries['country'] = countries.country.replace(name_change)
world = world[world.name.isin(countries.country.values)]

world = world.sort_values(by='name').reset_index()

countries = countries.sort_values(by='country').reset_index()

world = world.merge(countries, left_on=['name'], right_on=['country'])
prices = countries.columns[2:-2]

fig, ax = plt.subplots(len(prices), figsize=(16,6*len(prices)))



c = 0

for i in range(len(prices)):

    

    # some column names are repeated in the dataset, but the data is different.

    # An if-else makes sure each of these repeated columns in mapped.

    if type(world[prices[i]]) is pd.DataFrame:

        col = world[prices[i]].iloc[:,c]

        c -= 1

        c = abs(c)

    else:

        col = world[prices[i]] 

                              

    world.plot(column=col,

                ax=ax[i],

                legend=True,

                legend_kwds={'label': "Cost"})

    ax[i].title.set_text(prices[i])
data = world.iloc[:,9:]

x = data.values

min_max_scalar = preprocessing.MinMaxScaler()

x_scaled = min_max_scalar.fit_transform(x)

data_norm = pd.DataFrame(x_scaled)

data_norm.columns = data.columns
df_summary = pd.DataFrame(world['country'])

df_summary['total'] = data_norm.iloc[:,:56].mean(axis=1)
fig, ax = plt.subplots(1, figsize=(16,6))

                              

world.plot(column=df_summary['total'], ax=ax,

            legend=True,

            legend_kwds={'label': "Most to least expensive place to live"})

ax.title.set_text("All prices normalized for each Country")
df_summary.sort_values(by='total', ascending=False).head(3)