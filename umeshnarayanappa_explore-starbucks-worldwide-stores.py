import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

%matplotlib inline

sns.set(style="white", context="talk")



starbucks = pd.read_csv('../input/directory.csv')
from IPython.display import Image

Image(url='https://rde-stanford-edu.s3.amazonaws.com/Hospitality/Images/starbucks-header.jpg', embed = True)
starbucks.head(5)
starbucks.tail(5)
starbucks.shape
starbucks.notnull().sum()
starbucks.notnull().sum() * 100/starbucks.shape[0]
starbucks.shape[0]
len(starbucks.Country.unique())
starbucks.Country.value_counts().head(1)
starbucks.Country.value_counts().head(10)
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Top 10 Countries with Most Number of Starbucks Stores")

starbucks.Country.value_counts().head(10).plot(kind="bar", color = "maroon")
starbucks.City.value_counts().head(1)
starbucks.City.value_counts().head(10)
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Top 10 Cities with most number of Starbucks Stores")

starbucks.City.value_counts().head(10).plot(kind="bar")

plt.show()
starbucks['Ownership Type'].value_counts()
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Who owns the stores?")

starbucks['Ownership Type'].value_counts().plot(kind="bar", color = "maroon")

plt.show()
usa_states = starbucks[starbucks['Country'] == 'US']

usa_states['State/Province'].value_counts().head(1)
usa_states['State/Province'].value_counts().head(10)
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title="What are the Top 10 States in USA with most number of stores?")

usa_states['State/Province'].value_counts().head(10).plot(kind="bar")

plt.show()
starbucks.Brand.value_counts()
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title="Brand under which Starbucks Operates")

starbucks.Brand.value_counts().plot(kind="bar", color = "maroon")

plt.show()
plt.figure(figsize=(12,9))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()



m.drawmapboundary(fill_color='white')



x, y = m(list(starbucks["Longitude"].astype(float)), list(starbucks["Latitude"].astype(float)))

m.plot(x, y, 'bo', markersize = 5, alpha = 0.6, color = "blue")



plt.title('Starbucks Stores Across the World')

plt.show()
plt.figure(figsize=(10,8))

m = Basemap(projection='mill', llcrnrlat = 20, urcrnrlat = 50, llcrnrlon = -130, urcrnrlon = -60, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

m.drawmapboundary(fill_color='white')



x, y = m(list(usa_states["Longitude"].astype(float)), list(usa_states["Latitude"].astype(float)))

m.plot(x, y, 'bo', markersize = 5)



plt.title('Extinct and Endangered Languages in USA')

plt.show()