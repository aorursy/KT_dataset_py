import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm

from numpy import array
cities = pd.read_csv('../input/cities_r2.csv')

cities.head(2)
cities["lat"] = cities['location'].apply(lambda x: float(x.split(',')[0]))

cities["long"] = cities['location'].apply(lambda x: float(x.split(',')[1]))
def plot_map(sizes, colorbarValue):



    f, ax = plt.subplots(figsize=(12, 9))



    # initialize Basemap

    map = Basemap(width=5000000,

                  height=3500000,

                  resolution='l',

                  projection='aea',

                  llcrnrlon=69,

                  llcrnrlat=6,

                  urcrnrlon=99,

                  urcrnrlat=36,

                  lon_0=78,

                  lat_0=20,

                  ax=ax)



    # draw map boundaries

    map.drawmapboundary(fill_color='white')

    map.fillcontinents(color='#313438', lake_color='#313438', zorder=0.5)

    map.drawcountries(color='white')



    # show scatter point on map

    x, y = map(array(cities["long"]), array(cities["lat"]))

    cs = map.scatter(x, y, s=sizes, marker="o", c=sizes, cmap=cm.Dark2, alpha=0.5)



    # add colorbar.

    cbar = map.colorbar(cs, location='right',pad="5%")

    cbar.ax.set_yticklabels(colorbarValue)



    plt.show()
population_sizes = cities["population_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["population_total"].min(), cities["population_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
literacy_sizes = cities["literates_total"].apply(lambda x: int(x / 2000))

colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(literacy_sizes, colorbarValue)