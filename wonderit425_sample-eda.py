# basics

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



#misc

import gc

import time

import warnings



#viz

import matplotlib.pyplot as plt

import seaborn as sns 

import matplotlib.gridspec as gridspec 

import matplotlib.gridspec as gridspec 



# graph viz

import plotly.offline as pyo

from plotly.graph_objs import *

import plotly.graph_objs as go



#map section

import imageio

import folium

import folium.plugins as plugins

from mpl_toolkits.basemap import Basemap





#graph section

import networkx as nx

import heapq  # for getting top n number of things from list,dict





#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

warnings.filterwarnings("ignore")

pyo.init_notebook_mode()



%matplotlib inline
train = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/train.csv')

valid = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/valid.csv')



end_time=time.time()

print("Took",end_time-start_time,"s")
# take a peak

train.head(5)

#Get the distribution of the ratings

x=train['rating'].value_counts()

x=x.sort_index()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Review Rating Distribution")

plt.ylabel('# of reviews', fontsize=12)

plt.xlabel('Review Ratings ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()