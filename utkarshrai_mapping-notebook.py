%pip install geojsoncontour
%pip install openturns
import folium

import branca

from folium import plugins

import matplotlib.pyplot as plt

from scipy.interpolate import griddata

import geojsoncontour

import scipy as sp

import scipy.ndimage
import numpy as np

import openturns as ot

import matplotlib.pyplot as plt

import math

import pandas as pd



import scipy.interpolate as interpolate


#import numpy as np

#df = pd.read_csv('~/thoreau-backend/frontend/static/water/LabData/LAB_DATA_with_SENSOR_data_MAIN_for_ANALYSIS.csv')

df = pd.read_csv('../input/mappingtest/test.csv',error_bad_lines=False)

#df=df[df['Location']=='DELH']

print(df.keys())
df2=df[['CDOM [RFU]','GPS Lat.', 'GPS Long.']].dropna().round(3).drop_duplicates()

df2.head()
combined = np.vstack((df2['GPS Lat.'].values, df2['GPS Long.'].values)).T


coordinates = ot.Sample(combined)

observations = ot.Sample(df2['CDOM [RFU]'].values,1)



# Extract coordinates

x = np.array(coordinates[:,0])

y = np.array(coordinates[:,1])



# Plot the data with a scatter plot and a color map



fig = plt.figure()

plt.scatter(x, y, c=observations, cmap='seismic')

plt.colorbar()

plt.show()


x=df2.loc[:,'GPS Lat.']

y=df2.loc[:,'GPS Long.']



xi = np.linspace(x.min(),x.max(), 500)

yi = np.linspace(y.min(), y.max(), 500)

xi, yi = np.meshgrid(xi, yi)

Z=df2.loc[:,'CDOM [RFU]']

Z = np.array(Z)

Z = Z.astype(float)

    # calculating z values for whole grid 

zi = interpolate.griddata((x, y), Z, (xi, yi), method='cubic')
len(Z)
xi = np.linspace(x.min(),x.max(), 500)

yi = np.linspace(y.min(), y.max(), 500)

xi, yi = np.meshgrid(xi, yi)

Z=df2.loc[:,'CDOM [RFU]']

Z = np.array(Z)

Z = Z.astype(float)

    # calculating z values for whole grid 

zi = interpolate.griddata((x, y), Z, (xi, yi), method='linear')



#Creating contour plot with a step size of 1000

step_size=1000

cs = plt.contourf(xi,yi,zi,levels=20,cmap=plt.cm.jet)

plt.axis('off')

plt.savefig('pict.png', bbox_inches='tight', transparent=True)
from matplotlib.pyplot import imread

from folium import plugins

min_lon = y.min()

max_lon =y.max()

min_lat = x.min()

max_lat = x.max()



# create the map

map_ = folium.Map(location=[x.mean(), y.mean()],

                  tiles='openstreetmap', zoom_start = 13)



# read in png file to numpy array

data = imread('../input/picture/pict (1).png')



# Overlay the image

map_.add_children(folium.raster_layers.ImageOverlay(data, opacity=0.8, \

        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))

map_