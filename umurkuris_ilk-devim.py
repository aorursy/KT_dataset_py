import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import numpy as np

plt.style.use('fivethirtyeight')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins

from matplotlib import animation,rc

import io

import base64

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')

from scipy.misc import imread

import codecs

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

data=data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

data['casualities']=data['Killed']+data['Wounded']

data.head(3)
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Wounded.plot(kind = 'line', color = 'g',label = 'Wounded',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Killed.plot(color = 'r',label = 'Killed',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Killed', y='Wounded',alpha = 0.5,color = 'red')

plt.xlabel('Killed')              # label = name of label

plt.ylabel('Wounded')

plt.title('Killed Wounded Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Year.plot(kind = 'hist',bins = 50,figsize = (12,12)) #bin sütunların sayısısdır.

plt.show()