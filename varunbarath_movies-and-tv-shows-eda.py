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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
import math
import plotly.express as px
import plotly.graph_objects as go

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from plotly.colors import n_colors
from IPython.display import Image
from colorama import Fore, Back, Style
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL
custom_colors = ["#ff6b6b","#95d5b2","#a2d2ff","#72efdd"]
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors),size=1)
netflix_p = sns.light_palette(custom_colors[0], reverse=True)
sns.palplot(sns.color_palette(netflix_p),size=1)
hulu_p = sns.light_palette(custom_colors[1], reverse=True)
sns.palplot(sns.color_palette(hulu_p),size=1)
prime_p = sns.light_palette(custom_colors[2], reverse=True)
sns.palplot(sns.color_palette(prime_p),size=1)
disney_p = sns.dark_palette(custom_colors[3], reverse=True)
sns.palplot(sns.color_palette(disney_p),size=1)
df_tv = pd.read_csv('../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
df_movies = pd.read_csv('../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df_tv = df_tv.drop(['Unnamed: 0'], axis = 1) 
df_tv.head(5)
df_movies = df_movies.drop(['Unnamed: 0','ID'], axis = 1) 
df_movies.head(5)
len(df_movies['Directors'].unique())
len(df_movies['Genres'].unique())
def splitting(dataframe,col):
    result = dataframe[col].str.get_dummies(',')
    print('Done!')
    return result
m_genres = splitting(df_movies,'Genres')
m_lang = splitting(df_movies,'Language')
def val_sum(df,c):
    return df[c].sum(axis=0)
val_counts = []
dfs = [df_movies,df_tv]
cols = ['Netflix','Hulu','Prime Video','Disney+']

for x in dfs:
    for y in cols:
        val_counts.append(val_sum(x,y))
val_counts
def donut(i,df,sizes,title):
    plt.subplot(i)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True)

    centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(title)
    plt.axis('equal')
fig = plt.subplots(figsize=(16, 8))
labels = 'Netflix', 'Hulu','Prime','Disney+'
sizes1 = [val_counts[0], val_counts[1],val_counts[2],val_counts[3]]
sizes2 = [val_counts[4], val_counts[5],val_counts[6],val_counts[7]]
colors = custom_colors
explode = (0, 0, 0, 0) 

donut(121,df_movies,sizes1,'Movies')
donut(122,df_tv,sizes2,'TV shows')
plt.show()
def sunburst(dataframe,platform,c):
    dataframe=dataframe.loc[dataframe[platform] == 1]
    dataframe=dataframe.sort_values(by='IMDb', ascending=False)
    rating = dataframe[0:10]
    fig =px.sunburst(
    rating,
    path=['Title','Genres'],
    values='IMDb',
    color='IMDb',
    color_continuous_scale=c)
    fig.show()
sunburst(df_movies,'Netflix','amp')
sunburst(df_movies,'Hulu','Blugrn')
sunburst(df_movies,'Prime Video','haline')
sunburst(df_movies,'Disney+','dense')
def kde(i,dataframe,platform,c):
    plt.subplot(i)
    dataframe=dataframe.loc[dataframe[platform] == 1]
    sns.kdeplot(data=dataframe['Runtime'], color=custom_colors[c],shade=True)
    plt.xlabel('Runtime in minutes', fontsize = 15)
    plt.legend(fontsize = 15);
    plt.subplot(i+1)
    sns.kdeplot(data=dataframe['Year'], color=custom_colors[c],shade=True)
    plt.xlabel('Release Year', fontsize = 15)
    plt.legend(fontsize = 15);
plt.figure(figsize = (16, 8))

kde(421,df_movies,'Netflix',0)
kde(423,df_movies,'Hulu',1)
kde(425,df_movies,'Prime Video',2)
kde(427,df_movies,'Disney+',3)
df_t = df_tv.copy()
df_t = df_t[df_t['Age'].notna()]
df_t['Age']=df_t['Age'].str.replace('+','')
df_t['Age']=df_t['Age'].str.replace('all','0')
df_t['Age']=df_t['Age'].astype(str).astype(int)
def count_plot(i,dataframe,platform,p):
    plt.subplot(i)
    dataframe=dataframe.loc[dataframe[platform] == 1]
    ax = sns.countplot(x="Age", data=dataframe, palette=p, order=dataframe['Age'].value_counts().index[0:15])
    plt.xlabel('Age', fontsize = 15)
    plt.ylabel(platform, fontsize = 15)
    plt.legend(fontsize = 15);
plt.figure(figsize = (10, 9))

count_plot(421,df_t,'Netflix',netflix_p)
count_plot(423,df_t,'Hulu',hulu_p)
count_plot(425,df_t,'Prime Video',prime_p)
count_plot(427,df_t,'Disney+',disney_p)
r = df_movies.sort_values(by='IMDb', ascending=False)
r = r[0:20]
r = r[['Title','IMDb','Netflix','Hulu','Prime Video','Disney+']]
r.style.bar(subset=["Netflix",], color='#ff6b6b')\
                 .bar(subset=["Hulu"], color='#95d5b2')\
                 .bar(subset=["Prime Video"], color='#a2d2ff')\
                 .bar(subset=["Disney+"], color='#72efdd')\
                 .bar(subset=["IMDb"],color='#').background_gradient(cmap='Purples')
df_m = df_movies.copy()
df_m = df_m.dropna()

df_m['Rotten Tomatoes']=df_m['Rotten Tomatoes'].str.replace('%','')
df_m['Rotten Tomatoes']=df_m['Rotten Tomatoes'].astype(str).astype(int)
df_m['Directors']=df_m['Directors'].astype('str')
df_m=df_m.sort_values(by='Rotten Tomatoes', ascending=False)
rating = df_m[0:20]
sns.catplot(x="Rotten Tomatoes", y="Directors",data=rating, palette=netflix_p,height=7,kind="point")
df_l_merged = pd.concat([df_movies, m_lang], axis = 1, sort = False)
df_g_merged = pd.concat([df_movies, m_genres], axis = 1, sort = False)
def bar(dataframe,platform,c):
    dataframe=dataframe.loc[dataframe[platform] == 1]
    val_counts_l = dataframe.iloc[:,15:].sum(axis=0).sort_values(ascending=False)
    val_counts_lang = pd.DataFrame(val_counts_l,columns=['Number of movies'])
    return val_counts_lang[0:20].style.bar(subset=["Number of movies",], color=c)
bar(df_l_merged,'Netflix','#ff6b6b')
bar(df_g_merged,'Netflix','#ff6b6b')
bar(df_l_merged,'Hulu','#95d5b2')
bar(df_g_merged,'Hulu','#95d5b2')
bar(df_l_merged,'Prime Video','#a2d2ff')
bar(df_g_merged,'Prime Video','#a2d2ff')
bar(df_l_merged,'Disney+','#72efdd')
bar(df_g_merged,'Disney+','#72efdd')