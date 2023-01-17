# libraries

from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

 

# Always start witht the basemap function to initialize a map

m=Basemap()

 

# Then add element: draw coast line, map boundary, and fill continents:

m.drawcoastlines()

m.drawmapboundary()

m.fillcontinents()

 

# You can add rivers as well

m.drawrivers(color='#0000ff')

 

plt.show()
# Libraries

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

 

# Initialize the map

map = Basemap(llcrnrlon=-160, llcrnrlat=-60,urcrnrlon=160,urcrnrlat=70)

 

# Continent and countries!

map.drawmapboundary(fill_color='#A6CAE0')

map.fillcontinents(color='#e6b800',lake_color='#e6b800')

map.drawcountries(color="white")

plt.show()