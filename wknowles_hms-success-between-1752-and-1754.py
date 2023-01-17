from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon

# are the lat/lon values of the lower left and upper right corners

# of the map.

# lat_ts is the latitude of true scale.

# resolution = 'c' means use crude resolution coastlines.



m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=60,\

            llcrnrlon=-90,urcrnrlon=10,lat_ts=20,resolution='c')

m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

#m.drawcoastlines()

#m.fillcontinents(color='coral',lake_color='aqua')

# draw parallels and meridians.

#m.drawparallels(np.arange(-90.,91.,30.))

#m.drawmeridians(np.arange(-180.,181.,60.))

m.drawmapboundary(fill_color='aqua')

plt.title("Mercator Projection")

plt.show()
from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import animation

import pandas as pd



#read in data

shipdata = pd.read_csv('../input/CLIWOC15.csv')

lat = shipdata.Lat3

lon = shipdata.Lon3

coord=np.column_stack((list(lon),list(lat)))

ship=shipdata.ShipName

utc=shipdata.UTC

year=shipdata.Year

month=shipdata.Month

day=shipdata.Day



#take out lon/lat nan

utc=utc[~np.isnan(coord).any(axis=1)]

ship=ship[~np.isnan(coord).any(axis=1)]

year=year[~np.isnan(coord).any(axis=1)]

month=month[~np.isnan(coord).any(axis=1)]

day=day[~np.isnan(coord).any(axis=1)]

coord=coord[~np.isnan(coord).any(axis=1)]

data=np.column_stack((coord,ship,year,month,day,utc))



#find Success

np.count_nonzero(data[:,2]=='Success')

voyage=data[data[:,2]=='Success']



#sort time

voyage=voyage[voyage[:,6].argsort()]



#plot map

m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=60,\

            llcrnrlon=-90,urcrnrlon=10,lat_ts=20,resolution='l')

m.drawcoastlines()

#m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary(fill_color='aqua')



#draw path on the background

x,y=m(voyage[:,0],voyage[:,1])

m.plot(x,y,'.',color='grey',alpha=0.2)



#animation (based on https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/)



x,y = m(0, 0)

point = m.plot(x, y, 'o', markersize=7, color='red')[0]

def init():

    point.set_data([], [])

    return point,



def animate(i):

    x,y=m(cook[i][0],cook[i][1])

    point.set_data(x,y)

    plt.title('%d %d %d' % (cook[i][3],cook[i][4],cook[i][5]))

    return point,



output = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=355, interval=100, blit=True, repeat=False)



output.save('hmsSuccess.gif', writer='imagemagick')

plt.show()