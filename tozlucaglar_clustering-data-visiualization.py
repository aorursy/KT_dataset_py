# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import folium

from folium import plugins



print(folium.__version__)
import seaborn as sns # visualization library

import matplotlib.pyplot as plt # visualization library

import plotly.plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object



import warnings            

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.

plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.

# Any results you write to the current directory are saved as output.
itaksi = pd.read_csv("../input/binis_inis1.csv")

itaksi2 = pd.read_csv("../input/binis_inis2.csv")

print (itaksi.head())

print (itaksi2.head())
# itaksi2.info()

# country

print(itaksi2['ilce'].value_counts()[:10])

plt.figure(figsize=(22,10))

sns.countplot(itaksi2['ilce'])

plt.xticks(rotation=90)

plt.show()
m = folium.Map([45, 3], zoom_start=4)



plugins.ScrollZoomToggler().add_to(m)



#m.save(os.path.join('results', 'C:/Users/ctozluoglu/Desktop/shp/Plugins_1.html'))



m



# drop countries that are NaN

itaksi = itaksi[pd.isna(itaksi.POINT_Y)==False]

itaksi = itaksi[pd.isna(itaksi.POINT_X)==False]





data = np.array(

    [

        

        itaksi['POINT_Y'][:53000],  # Random latitudes in Europe.

        itaksi['POINT_X'][:53000],  # Random longitudes in Europe.

    ]

).T



m = folium.Map([41, 29], zoom_start=10)



plugins.FastMarkerCluster(data).add_to(m)



#m.save(os.path.join('results', 'C:/Users/ctozluoglu/Desktop/shp/Plugins_1.html'))

m