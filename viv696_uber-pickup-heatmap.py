import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



train_df = pd.read_csv("../input/train.csv")

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm







west, south, east, north = -74.26, 40.50, -73.70, 40.92

#west, south, east, north=-11.0,-11.0,-11.0,-11.0

fig = plt.figure(figsize=(14,14))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(train_df['pickup_longitude'].values, train_df['pickup_latitude'].values)

m.hexbin(x, y, gridsize=8000,

         bins='log', cmap=cm.YlOrRd_r);




