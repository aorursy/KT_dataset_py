import numpy as np

import pandas as pd

from os import listdir
listdir('/kaggle/input')
data = pd.read_csv('/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')
data.head()
data.drop(columns=set(data.columns)-{'Plot', 'Genre'}, inplace=True)

data.rename(str.lower, axis='columns', inplace=True)
data.info()
data['plot'] = data['plot'].map(str.lower)
genres_count = data.groupby('genre').size()

genres_count = genres_count[genres_count >= 100]

data = pd.merge(data, pd.DataFrame(genres_count).drop(columns=[0]), left_on='genre', right_index=True)
data.info()
data['genre'].unique()
from collections import ChainMap

defaults = dict(map(lambda genre: (genre, genre), data['genre'].unique()))

parser = dict(ChainMap({

    'romantic drama': 'romantic,drama',

    'crime drama': 'crime,drama',

    'comedy drama': 'comedy,drama',

    'romantic comedy': 'romantic,comedy',

    'musical comedy': 'musical,comedy',

    'comedy, drama': 'comedy,drama',

    'science fiction': 'sci-fi',

    'comedy-drama': 'comedy,drama',

    'romance': 'romantic'

}, defaults))

data['genre'] = data['genre'].map(parser)
data['genre'].unique()
from itertools import chain

categories = frozenset(chain.from_iterable(data['genre'].map(lambda genre: genre.split(','))))



genres = pd.DataFrame(dict(map(lambda category: (category, pd.Series([], dtype=np.uint8)), categories)),

                      columns=categories)

for index in data['genre'].unique():

    genres.loc[index] = np.array(list(map(set(index.split(',')).__contains__, categories)), dtype=np.uint8)

data = pd.merge(data, genres, left_on='genre', right_index=True).drop(columns=['genre'])
data.info()
data.to_csv('preprocessed-data.csv', index=False)
listdir('/kaggle/working')