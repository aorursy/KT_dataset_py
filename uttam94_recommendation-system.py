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
#Dataset url: https://grouplens.org/datasets/movielens/latest/

! ls ../input
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



text = ["London Paris London","Paris Paris London"]

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

count_matrix.toarray()
similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)
# importing the dataset

import pandas as pd

 

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

 

# this is a very toy example, do not try this at home unless you want to understand the usage differences

docs=["the house had a tiny little mouse",

      "the cat saw the mouse",

      "the mouse ran away from the house",

      "the cat finally ate the mouse",

      "the end of the mouse story"

     ]
 

#instantiate CountVectorizer()

cv=CountVectorizer()

 

# this steps generates word counts for the words in your docs

word_count_vector=cv.fit_transform(docs)

word_count_vector.toarray()
word_count_vector.shape
#compute IDF Values

 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(word_count_vector) 


# print idf values

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

 

# sort ascending

df_idf.sort_values(by=['idf_weights'])
#With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once. It’s really simple

from sklearn.feature_extraction.text import TfidfVectorizer#With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once. It’s really simple

tfidf_vectorizer=TfidfVectorizer(use_idf=True)

 

# just send in all your docs here

fitted_vectorizer=tfidf_vectorizer.fit(docs)

tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs)
import pandas as pd

import numpy as np
credits = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies_df = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
credits.head()
movies_df.head()
print("Credits:",credits.shape)

print("Movies Dataframe:",movies_df.shape)
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})

movies_df_merge = movies_df.merge(credits_column_renamed, on='id')

movies_df_merge.head()
movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])

movies_cleaned_df.head()
movies_cleaned_df.info()
movies_cleaned_df.head(1)['overview']
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3),

            stop_words = 'english')



# Filling NaNs with empty string

movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel



# Compute the sigmoid kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles

indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
indices
indices['Newlyweds']
sig[4799]
list(enumerate(sig[indices['Newlyweds']]))
def give_rec(title, sig=sig):

    # Get the index corresponding to original_title

    idx = indices[title]



    # Get the pairwsie similarity scores 

    sig_scores = list(enumerate(sig[idx]))



    # Sort the movies in the decending order on basis of similarity score 

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)



    # Scores of the 10 most similar movies

    sig_scores = sig_scores[1:11]



    # Movie indices

    movie_indices = [i[0] for i in sig_scores]



    # Top 10 most similar movies

    return movies_cleaned_df['original_title'].iloc[movie_indices]
# Testing our content-based recommendation system with the seminal film Spy Kids

give_rec('Avatar')
! ls ../input/movie-rating
import pandas as pd

import numpy as np
movies_df = pd.read_csv('../input/movie-rating/movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})

rating_df=pd.read_csv('../input/movie-rating/ratings.csv',usecols=['userId', 'movieId', 'rating'],

    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
movies_df.head()
rating_df.head()
#mering dataset on bbased on movie_id



df = pd.merge(rating_df,movies_df,on='movieId')

df.head()


combine_movie_rating = df.dropna(axis = 0, subset = ['title'])

movie_ratingCount = (combine_movie_rating.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'})

     [['title', 'totalRatingCount']]

    )

movie_ratingCount.head()
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')

rating_with_totalRatingCount.head()


print(movie_ratingCount['totalRatingCount'].describe())
popularity_threshold = 50

rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

rating_popular_movie.head()
rating_popular_movie.shape


## First lets create a Pivot matrix



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