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
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

import nltk

import plotly.express as px

from wordcloud import WordCloud
data = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")

data.head()
data.info()
data.shape
data.isna().sum()
rcParams["figure.figsize"] = 15,20

data["Genres"].value_counts()[:30].plot(kind="pie")
netflix_movies = data[data['Netflix'] == 1]

hulu_movies = data[data['Hulu'] == 1]

prime_movies =data[data['Prime Video'] == 1]

disney_movies = data[data['Disney+'] == 1]
rcParams["figure.figsize"] = 15,10

netflix_movies["Language"].value_counts()[:30].plot(kind="bar")
fig = px.histogram(data, x='Year',height=500, title='')

fig.show()
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
netflix_titles = netflix_movies.Title[:500].values

generate_word_cloud(netflix_titles)
prime_titles = prime_movies.Title[:500].values

generate_word_cloud(prime_titles)
hulu_titles = hulu_movies.Title[:500].values

generate_word_cloud(hulu_titles)
disney_titles = disney_movies.Title[:500].values

generate_word_cloud(disney_titles)
top_movies_IMDB=data[data['IMDb']==9.3][['Title','Directors']]
generate_word_cloud(top_movies_IMDB.Title)
top_Runtime = data.sort_values('Runtime',ascending = False).head(20)
fig = px.bar(top_Runtime, x='Title', y='Runtime', color='Runtime', height=500, title='Runtime of the top 10 longest movies')

fig.show()

yearly_movie_count = data.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
yearly_movie_count = yearly_movie_count.sort_values(by='MovieCount',ascending=False)
yearly_movie_count.head(30)
fig = px.bar(yearly_movie_count[:30], x='Year', y='MovieCount', color='MovieCount', height=600)

fig.show()