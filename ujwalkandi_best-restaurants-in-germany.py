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





print(gpd.__version__)
df = pd.read_excel("../input/best-restaurants-in-germany/Best Restaurants in Germany.xlsx")

df.head()
df.columns
print("Percentage null or na values in df")

((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
print("Rank   |   Chef   |   Restaurant | Stars")

for i in range(len(df)):

    print(df['Rank'].iloc[i].astype(int),"-", df['Chef'].iloc[i],"  -  ",df['Restaurant'].iloc[i],"-",df['Stars'].iloc[i])
df.City.value_counts().head(20)
city=df['City'].value_counts()

names=city.index

values=city.values
plt.figure(figsize=(9,8))

plt.style.use('default')

sns.barplot(x=names[:10],y=values[:10],edgecolor='k')

plt.xticks(rotation=90)

plt.ylabel('Restaurants Count')

plt.xlabel('City')

plt.title('Restaurants by City',fontsize=15)

plt.show()
plt.figure(figsize=(20,9))



plt.style.use('tableau-colorblind10')

df['City'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='orange', alpha=0.8)

  

for index, value in enumerate(df['City'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("City", fontsize=14)

plt.ylabel("Restaurants Count", fontsize=13)

plt.title("Restaurants by City", fontsize=15)

plt.legend()

plt.show()
df.Stars.value_counts()
stars_index=df.Stars.value_counts()

stars_index=stars_index.head(10)



plt.figure(figsize=(8,6))

ax=sns.barplot(x=stars_index.index,y=stars_index.values,palette=sns.cubehelix_palette(len(stars_index.index)))

plt.xlabel('Stars(rating)')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.title('Star rating of Restaurants')

plt.show()
df['Culinary Category'].value_counts()[:20]
rcParams['figure.figsize'] = 20,7

g = sns.countplot(x="Culinary Category",data=df, palette = "Set1")   

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Cuisines that are served across Germany',size = 15)

plt.show()
plt.figure(figsize=(20,9))



plt.style.use('default')

df['Culinary Category'].value_counts()[:15].plot(kind='bar',edgecolor='k',color='tomato', alpha=0.8)

  

for index, value in enumerate(df['Culinary Category'].value_counts()[:15]):

    plt.text(index, value, str(value))

plt.xlabel("Culinary Category", fontsize=14)

plt.ylabel("Count", fontsize=13)

plt.title("Cuisines that are served across Germany", fontsize=15)

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (8, 9)

plt.style.use('ggplot')



plt.subplot(2,2,1)

df['Open for Lunch'].value_counts().plot.bar(color = 'orange', width=0.5)

plt.title('Open for Lunch', fontsize = 12)



plt.subplot(2,2,2)

df['Open for Dinner'].value_counts().plot.bar(color = 'slateblue', width=0.5)

plt.title('Open for Dinner', fontsize = 12)



plt.subplot(2,2,3)

df['Open for Midday'].value_counts().plot.bar(color = 'orangered', width=0.5)

plt.title('Open for Midday', fontsize = 12)



plt.subplot(2,2,4)

df['Plant Holidays'].value_counts().plot.bar(color = 'chartreuse', width=0.5)

plt.title('Plant Holidays', fontsize = 12)



plt.show()
df['Quality Score'].value_counts().head(20)
fig, ax = plt.subplots(figsize=[16,4])

sns.distplot(df['Quality Score'],bins=50, kde=True,color='red')

ax.set_title('Quality Score of Restaurants')

ax. set_yticks([])

plt.show()
fig, ax = plt.subplots(figsize=[8,4])

sns.violinplot(data=df['Quality Score'], palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set_title('Quality Score of Restaurants')

ax.grid('off')

plt.show()
df['Price Per Person']= df['Price Per Person'].str.split("€", n = 1, expand = True) 

df['Price Per Person']
df.dropna(how='any',inplace=True)

df.head()
df['Price Per Person'].value_counts()
sns.countplot(df['Price Per Person']).set_xticklabels(sns.countplot(df['Price Per Person']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(20,6)

plt.title('Cost of Restuarant')

plt.show()
fig, ax = plt.subplots(figsize=[16,4])

sns.distplot(df['Price Per Person'],bins=50, kde=True,color='red', ax=ax)

ax.set_title('Cost Distrubution for all restaurants')





ax. set_yticks([])

ax2 = ax.twinx() # create a second y axis

y_max = df['Price Per Person'].astype(float).max() # maximum of the array



# find the percentages of the max y values.

# This will be where the "0%, 25%" labels will be placed

ticks = [0, 0.25*y_max, 0.5*y_max, 0.75*y_max, y_max] 



ax2.set_ylim(ax.get_ylim()) # set second y axis to have the same limits as the first y axis

ax2.set_yticks(ticks) 

# ax2.set_yticklabels(["0%", "25%","50%","75%","100%"]) # labels in percentage

ax2.grid("off")



plt.show()

plt.figure(figsize=(15,7))

chains=df['Restaurant'].value_counts()[:20]

sns.barplot(x=chains,y=chains.index,palette='Set1')

plt.title("Restaurant chains with most number of outlets",size=15,pad=20)

plt.xlabel("Number of outlets",size=15)

plt.show()
df.Courses.value_counts()[:20]
df['Courses']= df['Courses'].str.split(" ", n = 1, expand = True) 



df['Courses']
plt.figure(figsize=(12,6))

plt.style.use('seaborn-white')

sns.distplot(df['Courses'].astype(float),kde = False, color='r')

plt.ylabel('Count')

plt.title('Number of Courses served')

plt.show()