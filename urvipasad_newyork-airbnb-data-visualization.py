### Import packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')                  # To apply seaborn whitegrid style to the plots.

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_columns = 20 ##set column widths to display the data
NewyorkCityDf = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv') ## read the CSV file from kaggle
NewyorkCityDf.shape
NewyorkCityDf.describe()
NewyorkCityDf.info()
hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>1500][['name','host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False)

print(hostname_DF)
## Find the top 10 expensive Host's listing and host names



##NewyorkCityDf.setIndex(['host_name'])

hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>1500][['host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False).plot(kind = 'bar', figsize = (12,5))

plt.xlabel('host names')

plt.ylabel('price')

##hostname_DF.set_index('host_name')

print(hostname_DF)

##NewyorkCityDf.loc[NewyorkCityDf.price>1500][['host_name', 'price']][:11].sort_values(by = 'price', ascending = False).plot(kind = 'bar', xticks = 'host_name')


hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>8000][['name', 'price', 'neighbourhood_group']][:11].sort_values(by = 'price', ascending = False)

plt.figure(figsize=(12,7))

sns.barplot(y="name", x="price", data=hostname_DF, hue = 'neighbourhood_group',palette= 'gist_earth')
cheapesthostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price<50][['name', 'price', 'neighbourhood_group']][:11].sort_values(by = 'price')

plt.figure(figsize=(12,7))

ax = sns.barplot(y="name", x="price", data=cheapesthostname_DF, hue = 'neighbourhood_group',palette= 'gist_earth')
NewyorkCityDf.loc[NewyorkCityDf.price>8000][['host_name', 'price']][:11].sort_values(by = 'price', ascending = False)
NewyorkCityDf.columns.values.tolist()
## Hom many different types of rooms types are there in airbnb hosted for newyork

# Using matplotlib to add labels and title to the plot. 

plt.figure(figsize=(12,7))

plt.xlabel('Room Types')

plt.ylabel('Number of Items')

plt.title('Bar Chart showing the Number of rooms in each room type value')



# In order to save your plot into an image on your system, use the following command.

# The image will be saved in the directory of this notebook.

sns.countplot(x='room_type',  data=NewyorkCityDf)
## Hom many different types of rooms types are there in airbnb hosted for newyork

NewyorkCityDf['neighbourhood_group'].value_counts().sort_values().plot(kind = 'bar',colormap='BrBG', figsize=(12,5), fontsize = 15) 

# Using matplotlib to add labels and title to the plot. 

# Pandas and matplotlib are linked with each other in the notebook by the use of this line in the Imports: %matplotlib inline



plt.xlabel('Neighbourhood Group', fontsize=15)

plt.ylabel('Number of listing', fontsize=15)

plt.title('Bar Chart showing the Number of listing per neighbourhood_group', fontsize=15)

roomtypecount = pd.Series(NewyorkCityDf.groupby(['neighbourhood_group'])['room_type'].value_counts())
roomtypecount
NewyorkCityDf.groupby(['neighbourhood_group'])['room_type'].value_counts().sort_values().plot(kind = 'bar', figsize=(12,5), colormap = 'Dark2')
NewyorkCityDf.groupby(['neighbourhood_group'])['price'].mean().plot(kind = 'bar', figsize=(12,5))
NewyorkCityDf.groupby(['neighbourhood_group','room_type'])['price'].mean().sort_values(ascending = False)
plt.figure(figsize=(12,8))

ytickrange = np.arange(0, 14000, 500) 

ax = sns.countplot(x='room_type', hue="neighbourhood_group", data=NewyorkCityDf)

ax.set_yticks(ytickrange)
def groupPrice(price):

    if price < 100:

        return "Low Cost"

    elif price >=100 and price < 200:

        return "Middle Cost"

    else:

        return "High Cost"

      

price_group = NewyorkCityDf['price'].apply(groupPrice)

NewyorkCityDf.insert(10, "price_group", price_group, True)

NewyorkCityDf.head(5)
g = sns.catplot(x="neighbourhood_group", hue="room_type",col="price_group", data=NewyorkCityDf, kind="count", height=5, aspect=1)

plt.show()
BBox = ((NewyorkCityDf.longitude.min(),NewyorkCityDf.longitude.max(),NewyorkCityDf.latitude.min(),NewyorkCityDf.latitude.max()))
BBox
newyorkMap = plt.imread("/kaggle/input/nycjpg/NYC_UV.jpg")
import folium

folium_map = folium.Map(location=[40.738, -73.98],

                        zoom_start=13,

                        tiles="CartoDB dark_matter")



folium.CircleMarker(location=[40.738, -73.98],fill=True).add_to(folium_map)

folium_map
ig, ax = plt.subplots(figsize = (18,20))

ax = sns.scatterplot(data=NewyorkCityDf, x='longitude', y='latitude', hue='neighbourhood_group')

##(NewyorkCityDf.longitude, NewyorkCityDf.latitude, zorder=1, alpha= 0.5, c='b', s=10)

ax.set_title('Plotting Spatial Data on Newyork Map')

ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])

ax.imshow(newyorkMap, zorder=0, extent = BBox, aspect= 'equal', alpha = 0.5, cmap  = 'winter')
plt.figure(figsize=(12,6))

sns.barplot(y="neighbourhood", x="price", data=NewyorkCityDf.nlargest(10,['price']))

plt.ioff()
data=NewyorkCityDf.nlargest(10,['price'])
data