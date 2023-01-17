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
import numpy as np

import pandas as pd

from tabulate import tabulate

import matplotlib.pyplot as plt

from matplotlib import rcParams

import matplotlib

import seaborn as sns

import geopandas as gpd

from geopandas.tools import geocode

import squarify    # pip install squarify (algorithm for treemap)

from mpl_toolkits.basemap import Basemap

import folium

import plotly.express as px

import plotly.graph_objects as go
rwgor=pd.read_csv('../input/tripadvisor-attractions-reviews-nearby-locations/attractionReviewComments_GOR.csv', engine='python')

rwgor.head()
rwscc=pd.read_csv('../input/tripadvisor-attractions-reviews-nearby-locations/attractionReviewComments_SCC.csv', engine='python')

rwscc.head()
rwgor['Trip type'].value_counts()
fig = go.Figure(data=[go.Pie(labels=['COUPLES', 'FAMILY', 'FRIENDS', 'SOLO', 'BUSINESS'], values=rwgor['Trip type'].value_counts(),textinfo='label+percent',title='Great Ocean Road (GOR)')])

fig.show()
rwscc['Trip type'].value_counts()
fig = go.Figure(data=[go.Pie(labels=['COUPLES', 'FAMILY', 'FRIENDS', 'SOLO', 'BUSINESS'], values=rwscc['Trip type'].value_counts(),textinfo='label+percent',title='Sunshine Coast (SCC)')])

fig.show()
new = rwgor['reviewer hometown'].str.split(",", n = 1, expand = True) 

  

# making separate first name column from new data frame 

rwgor["places"]= new[0] 

  

# making separate last name column from new data frame 

rwgor["countries"]= new[1]
new = rwscc['reviewer hometown'].str.split(",", n = 1, expand = True) 

  

# making separate first name column from new data frame 

rwscc["places"]= new[0] 

  

# making separate last name column from new data frame 

rwscc["countries"]= new[1]
rwgor["places"].value_counts().head(20)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwgor['places'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='royalblue', alpha=0.8)

  

for index, value in enumerate(rwgor['places'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("City-Region", fontsize=14)

plt.yscale('log')

plt.ylabel("Count", fontsize=13)

plt.title("Reviews by City-Region(Great Ocean Road)", fontsize=15)

plt.legend()

plt.show()
rwscc["places"].value_counts().head(20)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwscc['places'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='orange', alpha=0.8)

  

for index, value in enumerate(rwscc['places'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("City-Region", fontsize=14)

plt.yscale('log')

plt.ylabel("Count", fontsize=13)

plt.title("Reviews by City-Region(Sunshine Coast)", fontsize=15)

plt.legend()

plt.show()
rwgor["countries"].value_counts().head(10)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwgor['countries'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='royalblue', alpha=0.8)

  

for index, value in enumerate(rwgor['countries'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("Country", fontsize=14)

plt.yscale('log')

plt.ylabel("Count", fontsize=13)

plt.title("Reviews by Country(Great Ocean Road)", fontsize=15)

plt.legend()

plt.show()
rwscc["countries"].value_counts().head(10)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwscc['countries'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='orange', alpha=0.8)

  

for index, value in enumerate(rwscc['countries'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("Country", fontsize=14)

plt.yscale('log')

plt.ylabel("Count", fontsize=13)

plt.title("Reviews by Country(Sunshine Coast)", fontsize=15)

plt.legend()

plt.show()
rwgor['reviewer rating'].value_counts()
rating_index=rwgor['reviewer rating'].value_counts()

rating_index=rating_index.head(10)



plt.figure(figsize=(8,8))

ax=sns.barplot(x=rating_index.index,y=rating_index.values,palette=sns.cubehelix_palette(len(rating_index.index)))

plt.xlabel('Traveller rating', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(rotation=90)

plt.title('Traveller Rating(Great Ocean Road)', fontsize=15)

plt.show()
rwscc['reviewer rating'].value_counts()
rating_index=rwscc['reviewer rating'].value_counts()

rating_index=rating_index.head(10)



plt.figure(figsize=(8,8))

ax=sns.barplot(x=rating_index.index,y=rating_index.values,palette=sns.cubehelix_palette(len(rating_index.index)))

plt.xlabel('Traveller rating', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(rotation=90)

plt.title('Traveller Rating(Sunshine Coast)', fontsize=15)

plt.show()
rwgor.attractionName.value_counts().head(30)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwgor['attractionName'].value_counts()[:20].plot(kind='bar',edgecolor='k',color='royalblue', alpha=0.8)

  

for index, value in enumerate(rwgor['attractionName'].value_counts()[:20]):

    plt.text(index, value, str(value))

plt.xlabel("Attraction Name", fontsize=14)

plt.yscale('log')

plt.ylabel("Reviews Count", fontsize=13)

plt.title("Reviews on Attractions located across the Great Ocean Road", fontsize=15)

plt.legend()

plt.show()
rwscc.attractionName.value_counts().head(30)
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

rwscc['attractionName'].value_counts()[:20].plot(kind='bar',edgecolor='k',color='orange', alpha=0.8)

  

for index, value in enumerate(rwscc['attractionName'].value_counts()[:20]):

    plt.text(index, value, str(value))

plt.xlabel("Attraction Name", fontsize=14)

plt.yscale('log')

plt.ylabel("Reviews Count", fontsize=13)

plt.title("Reviews on Attractions located across the Sunshine Coast", fontsize=15)

plt.legend()

plt.show()
rwgor.attractionName.value_counts().head(20)
rwgor['attraction score summary'].value_counts()
fig = px.histogram(rwgor, x="attractionName", y="attraction score summary", color="attraction score summary", 

                   title="Rating of Attractions across the Great Ocean Road", 

                   labels={'attractionName':'Attraction Name','attraction score summary':'Rating','sum of attraction score summary':'Reviews Count'})

fig.show()
rwscc.attractionName.value_counts().head(20)
rwscc['attraction score summary'].value_counts()
fig = px.histogram(rwscc, x="attractionName", y="attraction score summary", color="attraction score summary", 

                   title="Rating of Attractions across the Sunshine Coast",

                   labels={'attractionName':'Attraction Name','attraction score summary':'Rating','sum of attraction score summary':'Reviews Count'})

fig.show()