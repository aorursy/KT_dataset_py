import pandas as pd

import matplotlib.pyplot as plt



movies = pd.read_csv('../input/movie_metadata.csv')

print(movies.head())

movies = movies[['duration', 'gross', 'budget', 'imdb_score', 'movie_facebook_likes']]

print(movies.head())
elements = ['duration', 'gross', 'budget', 'imdb_score', 'movie_facebook_likes']

for id1 in range(5):

    for id2 in range(id1+1,5):

        element1 = elements[id1]

        element2 = elements[id2]

        plt.plot(movies[element1], movies[element2], '.')

        plt.xlabel(element1)

        plt.ylabel(element2)

        plt.show()

        