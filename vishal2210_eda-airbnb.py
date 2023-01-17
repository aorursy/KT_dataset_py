# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import plotly.express as px

import geopandas as gpd





import urllib

import folium

from folium.plugins import HeatMap

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
data.info()
# Correlation between columns (number_of_reviews and reviews_per_month show highest correlation)

data.corr().style.background_gradient(cmap='coolwarm')
# Room types count

plt.figure(figsize=(10,10))

sns.countplot(data.sort_values('room_type').room_type,palette='Set1')

plt.title('Room_type_count')

plt.xlabel('Room_type')

plt.ylabel('Count')

plt.show()
data['room_type'].unique()
# Price and availability based on room type

f, ax = plt.subplots(figsize=(10, 10))

sns.despine(f, left=True, bottom=True)

room_ranking = ['Private room', 'Entire home/apt', 'Shared room']

sns.scatterplot(x="availability_365", y="price",

                hue="room_type", 

                palette="ch:r=-.2,d=.3_r",

                hue_order=room_ranking,

                sizes=(1, 8), linewidth=0,

                data=data, ax=ax)
# Comparing numerical data 

plt.figure(figsize=(20,20))

sns.pairplot(data[['price','minimum_nights','number_of_reviews','reviews_per_month',

                   'calculated_host_listings_count','availability_365','room_type']], hue="room_type")
#Distribution of availability_365 with availability less than 100

plt.figure(figsize=(14,8))

sns.distplot(data[data.availability_365>100].availability_365)

plt.title('Distribution of availability_365 (only where availability_365<100)')

plt.show()
# Neighbourhood group and price

plt.figure(figsize=(14,8))

sns.boxplot(data.neighbourhood_group,data.price)

# sns.swarmplot(data.neighbourhood_group,data.price)

plt.ylim(0,2000)

plt.show()
# Heat map for locations used in dataset

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
#Neighbourhood groups

plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=data)
#Neighbourhood groups with price less than 100

sub_6=data[data.price < 100]

viz_4=sub_6.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',

                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))

viz_4.legend()
plt.figure(figsize=(10,8))

#loading the png NYC image found on Google and saving to my local folder along with the project

# i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')

nyc_img=cv2.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png')

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()

#using scatterplot again

sub_6.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax, 

           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)

plt.legend()

plt.show()
#Room types and associated prices(Size of bubbles)

fig = px.scatter_mapbox(data, 

                        lat="latitude", 

                        lon="longitude", 

                        color="room_type", 

                        size="price",

                        color_continuous_scale=px.colors.cyclical.IceFire, 

                        size_max=30, 

                        opacity = .70,

                        zoom=11,

                       )

fig.layout.mapbox.style = 'carto-positron'

fig.update_layout(title_text = 'Room types and prices<br>(Click legend to toggle room types)', height = 800)



fig.show()
# Price and number of reviews

fig = px.scatter_3d(data, x="latitude", y="longitude", z="price", color="room_type", size="number_of_reviews", hover_name="name",

                   color_discrete_map = {"latitude": "blue", "longitude": "green", "price":"red"})

fig.show()

#symbol="result",