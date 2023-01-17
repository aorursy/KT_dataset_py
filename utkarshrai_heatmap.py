# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

import scipy.interpolate as interpolate

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

#import Tkinter, tkFileDialog, tkMessageBox

from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

df = pd.read_csv('../input/soil-map2/123.csv',encoding="ISO-8859-1", error_bad_lines = False)

df= DataFrame(df, columns=['Lat','Long','Volumetric Water Content'])

df = df.dropna()

df = df.drop_duplicates(keep='first')



Y = df['Lat']

X = df['Long']

X = df['Long']*-1

Z = df['Volumetric Water Content']





X = np.array(X)

Y = np.array(Y)

Z = np.array(Z)





yi = np.linspace(Y.min(),Y.max(), 550)

xi = np.linspace(X.min(), X.max(), 550)

xi, yi = np.meshgrid(xi, yi)

rbf = interpolate.Rbf( X, Y, Z,method='cubic')



zi1 = interpolate.griddata( (X, Y), Z, (xi, yi),method='cubic')



zi2=rbf(xi,yi)



ind = np.isnan(zi1)

zi2[ind]=np.nan



Z_range = (0.2, 0.45)



plt.figure()

plt.subplots_adjust(left=0,right=1,bottom=0,top=1)

CS = plt.contourf(xi,yi,zi2,50, vmin=0.2, vmax=0.45, cmap='viridis')   #jet for normal, jet_r for reverse

#plt.colorbar()



Y_range = plt.gca().get_ylim()                         # x min value and x max value

X_range = plt.gca().get_xlim()

# data points 

plt.gca().set_ylim(Y_range[0], Y_range[1])

plt.gca().set_xlim(X_range[0], X_range[1])

plt.axis('off')  

plt.savefig( "contour_plot.png" , bbox_inches='tight', transparent=True)  #"Documents/Rajmundry_Project/RIVER_DATA/" + 
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from datetime import datetime

from pandas import DataFrame

import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pyplot, transforms





df = pd.read_csv('../input/soil-type/Soil_map.csv',encoding="ISO-8859-1", error_bad_lines = False)

df= DataFrame(df, columns=['Lat','Long'])

df['Long'] = df['Long']*-1

y1 = df['Lat']

x1 = df['Long']



#fig = plt.figure()

#fig,pyplot.subplots(figsize=(16, 6))

pyplot.xlim(y1.min(), y1.max())

pyplot.xlim(x1.min(), x1.max())

line = plt.plot(x1,y1, '.', markersize=1, color = "#000000")

plt.axis('off')

plt.savefig( "contour_lines.png" , bbox_inches='tight', transparent=True)
import folium

from matplotlib.pyplot import imread

from folium import plugins

min_lon = X.min()

max_lon =X.max()

min_lat = Y.min()

max_lat = Y.max()

token = "pk.eyJ1IjoicHlzZWlkb24iLCJhIjoiY2p6bnFxYWZ2MDcxNDNtcnNkMGpoeTRneSJ9.rofxOMYv1GxYQas955YTJg" # your mapbox token

tileurl = 'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token=' + str(token)



# create the map

map_ = folium.Map(location=[Y.mean(), X.mean()],

                  tiles=tileurl, attr='Mapbox', zoom_start = 16)



# read in png file to numpy array

data = imread('../input/mapscon/contour_plot (4).png')



# Overlay the image

map_.add_children(folium.raster_layers.ImageOverlay(data, opacity=0.8, \

        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))



data2 = imread('../input/mapscon/contour_lines (7).png')

#y1*=-1

min_lon = x1.min()

max_lon =x1.max()

min_lat = y1.min()

max_lat = y1.max()



# Overlay the image

map_.add_children(folium.raster_layers.ImageOverlay(data2, opacity=0.8, \

        bounds =[[min_lat-0.00025, min_lon], [max_lat-0.00025, max_lon]]))

map_
import matplotlib.pyplot as plt

import matplotlib as mpl



# Make a figure and axes with dimensions as desired.

fig = plt.figure(figsize=(8, 2))

ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])



# Set the colormap and norm to correspond to the data for which

# the colorbar will be used.

cmap = mpl.cm.viridis

norm = mpl.colors.Normalize(vmin=0.25, vmax=0.4)



# ColorbarBase derives from ScalarMappable and puts a colorbar

# in a specified axes, so it has everything needed for a

# standalone colorbar.  There are many more kwargs, but the

# following gives a basic continuous colorbar with ticks

# and labels.

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,

                                norm=norm,

                                orientation='horizontal')

mpl.rcParams.update({'font.size': 24})

cb1.set_label('Some Units')







plt.show()
y1.max()