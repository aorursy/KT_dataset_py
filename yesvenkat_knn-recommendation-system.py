# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
movies_df = pd.read_csv('/kaggle/input/movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})

rating_df=pd.read_csv('/kaggle/input/ratings.csv',usecols=['userId', 'movieId', 'rating'],

    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
movies_df.head()
rating_df.head()
df = pd.merge(rating_df,movies_df,on='movieId')

df.head()
combine_movie_rating = df.dropna(axis = 0, subset = ['title'])

movie_ratingCount = (combine_movie_rating.

     groupby(by = ['title'])['rating'].

     count().

     reset_index().

     rename(columns = {'rating': 'totalRatingCount'})

     [['title', 'totalRatingCount']]

    )

movie_ratingCount.head()
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')

rating_with_totalRatingCount.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print(movie_ratingCount['totalRatingCount'].describe())
popularity_threshold = 50

rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

rating_popular_movie.head()
rating_popular_movie.shape



movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)

movie_features_df.head()
from scipy.sparse import csr_matrix



movie_features_df_matrix = csr_matrix(movie_features_df.values)



from sklearn.neighbors import NearestNeighbors





model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(movie_features_df_matrix)
movie_features_df.shape

query_index = np.random.choice(movie_features_df.shape[0])

print(query_index)

distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
movie_features_df.head()

for i in range(0, len(distances.flatten())):

    if i == 0:

        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))

    else:

        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))