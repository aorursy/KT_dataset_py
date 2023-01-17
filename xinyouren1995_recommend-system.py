

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Load data for all movies

movie_titles = pd.read_csv('../input/netflix-prize-data/movie_titles.csv', 

                           encoding = 'ISO-8859-1', 

                           header = None, 

                           names = ['Id', 'Year', 'Name']).set_index('Id')



print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))

movie_titles.sample(5)
# Load a movie metadata dataset

movie_metadata = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()

# Remove the long tail of rarly rated moves

movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)



print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))

movie_metadata.sample(5)
from collections import deque
# Load single data-file

df_raw = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', 

                     header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])





# Find empty rows to slice dataframe for each movie

tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()

movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]



# Shift the movie_indices by one to get start and endpoints of all movies

shifted_movie_indices = deque(movie_indices)

shifted_movie_indices.rotate(-1)





# Gather all dataframes

user_data = []



# Iterate over all movies

for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

    

    # Check if it is the last movie in the file

    if df_id_1<df_id_2:

        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()

    else:

        tmp_df = df_raw.loc[df_id_1+1:].copy()

        

    # Create movie_id column

    tmp_df['Movie'] = movie_id

    

    # Append dataframe to list

    user_data.append(tmp_df)



# Combine all dataframes

df = pd.concat(user_data)

del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

print('Shape User-Ratings:\t{}'.format(df.shape))

df.sample(5)
# Get data

data_year = movie_titles['Year'].value_counts().sort_index()
data_year.tail()
plt.figure(figsize=(8,6))

plt.plot(data_year.index, data_year.values)

plt.xlabel('Release Year')

plt.ylabel('Moives')

plt.title('{} Movies Grouped By Year Of Release'.format(movie_titles.shape[0]))

plt.grid()

plt.show()
data_rating = df['Rating'].value_counts().sort_index(ascending=False)
x = [5,4,3,2,1]

s=['{:.1f} %'.format(val) for val in (data_rating.values / df.shape[0] * 100)]

plt.figure(figsize=(8,6))

plt.bar(data_rating.index,data_rating.values)

for i in range(5):

    plt.text(x[i],data_rating.values[i],s=s[i])

plt.title('Distribution Of {} Netflix-Ratings'.format(df.shape[0]))

plt.xlabel('Rating')

plt.ylabel('Count')

plt.grid(which='minor', axis='y')

plt.show()
##### Ratings Per Movie #####

# Get data

data_movie = df.groupby('Movie')['Rating'].count().clip(upper=9999)



plt.figure(figsize=(8,6))

plt.hist(x = data_movie.values,bins = 100)

plt.title('Distribution Of Ratings Per Movie (Clipped at 9999)')

plt.xlabel('Ratings Per Movie')

plt.ylabel('Count')

plt.grid(axis='y')

plt.show()


##### Ratings Per User #####

# Get data

data_user = df.groupby('User')['Rating'].count().clip(upper=199)



plt.figure(figsize=(8,6))

plt.hist(x = data_user.values,bins = 100)

plt.title('Distribution Of Ratings Per User (Clipped at 199)')

plt.xlabel('Ratings Per User')

plt.ylabel('Count')

plt.grid(axis='y')

plt.show()



# Filter sparse movies

min_movie_ratings = 10000

filter_movies = (df['Movie'].value_counts()>min_movie_ratings)

filter_movies = filter_movies[filter_movies].index.tolist()



# Filter sparse users

min_user_ratings = 200

filter_users = (df['User'].value_counts()>min_user_ratings)

filter_users = filter_users[filter_users].index.tolist()



# Actual filtering

df_filterd = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]

del filter_movies, filter_users, min_movie_ratings, min_user_ratings

print('Shape User-Ratings unfiltered:\t{}'.format(df.shape))

print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))
# Shuffle DataFrame

df_filterd = df_filterd.drop('Date', axis=1).sample(frac=1).reset_index(drop=True)



# Testingsize

n = 100000



# Split train- & testset

df_train = df_filterd[:-n]

df_test = df_filterd[-n:]
# Create a user-movie matrix with empty values

df_p = df_train.pivot_table(index='User', columns='Movie', values='Rating')

print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

df_p.sample(3)
from sklearn.metrics import mean_squared_error
# Top n movies

n = 10



# Compute mean rating for all movies

ratings_mean = df_p.mean(axis=0).sort_values(ascending=False).rename('Rating-Mean').to_frame()



# Count ratings for all movies

ratings_count = df_p.count(axis=0).rename('Rating-Count').to_frame()



# Combine ratings_mean, ratings_count and movie_titles

ranking_mean_rating = ratings_mean.head(n).join(ratings_count).join(movie_titles.drop('Year', axis=1))





# Join labels and predictions

df_prediction = df_test.set_index('Movie').join(ratings_mean)[['Rating', 'Rating-Mean']]

y_true = df_prediction['Rating']

y_pred = df_prediction['Rating-Mean']



# Compute RMSE

rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))



plt.figure(figsize=(10,6))

plt.barh(list(range(1, n+1)),ranking_mean_rating['Rating-Mean'])

plt.title('Ranking Of Top {} Mean-Movie-Ratings: {:.4f} RMSE'.format(n, rmse))

plt.xlabel('Mean-Rating')

plt.xlim(4.3,4.6)

plt.ylabel('Movie')

for i in range(0,n):

    plt.text(ranking_mean_rating['Rating-Mean'].values[i],i+1,

             ranking_mean_rating['Name'].values[i]+':'+

             ranking_mean_rating['Rating-Mean'].values.round(3)[i].astype(str),

            color='red')

plt.show()





# Number of minimum votes to be considered

m = 1000



# Mean rating for all movies

C = df_p.stack().mean()



# Mean rating for all movies separatly

R = df_p.mean(axis=0).values



# Rating count for all movies separatly

v = df_p.count().values





# Weighted formula to compute the weighted rating

weighted_score = (v/ (v+m) *R) + (m/ (v+m) *C)

# Sort ids to ranking

weighted_ranking = np.argsort(weighted_score)[::-1]

# Sort scores to ranking

weighted_score = np.sort(weighted_score)[::-1]

# Get movie ids

weighted_movie_ids = df_p.columns[weighted_ranking]





# Join labels and predictions

df_prediction = df_test.set_index('Movie').join(pd.DataFrame(weighted_score, index=weighted_movie_ids, columns=['Prediction']))[['Rating', 'Prediction']]

y_true = df_prediction['Rating']

y_pred = df_prediction['Prediction']



# Compute RMSE

rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))





# Create DataFrame for plotting

df_plot = pd.DataFrame(weighted_score[:n], columns=['Rating'])

df_plot.index = weighted_movie_ids[:10]

ranking_weighted_rating = df_plot.join(ratings_count).join(movie_titles)

del df_plot

plt.figure(figsize=(10,6))

plt.barh(list(range(1, n+1)),ranking_weighted_rating['Rating'])

plt.title('Ranking Of Top {} Weighted-Movie-Ratings: {:.4f} RMSE'.format(n, rmse))

plt.xlabel('Weighted Rating')

plt.xlim(4.15,4.6)

plt.ylabel('Movie')

for i in range(0,n):

    plt.text(ranking_weighted_rating['Rating'].values[i],i+1,

              ranking_weighted_rating['Name'].values[i]+':'+

              ranking_weighted_rating['Rating'].values.round(3)[i].astype(str),

              color='red')

plt.show()
# To create deep learning models

from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout

from keras.models import Model
# Create user- & movie-id mapping

user_id_mapping = {id:i for i, id in enumerate(df_filterd['User'].unique())}

movie_id_mapping = {id:i for i, id in enumerate(df_filterd['Movie'].unique())}





# Create correctly mapped train- & testset

train_user_data = df_train['User'].map(user_id_mapping)

train_movie_data = df_train['Movie'].map(movie_id_mapping)



test_user_data = df_test['User'].map(user_id_mapping)

test_movie_data = df_test['Movie'].map(movie_id_mapping)





# Get input variable-sizes

users = len(user_id_mapping)

movies = len(movie_id_mapping)

embedding_size = 10





##### Create model

# Set input layers

user_id_input = Input(shape=[1], name='user')

movie_id_input = Input(shape=[1], name='movie')



# Create embedding layers for users and movies

user_embedding = Embedding(output_dim=embedding_size, 

                           input_dim=users,

                           input_length=1, 

                           name='user_embedding')(user_id_input)

movie_embedding = Embedding(output_dim=embedding_size, 

                            input_dim=movies,

                            input_length=1, 

                            name='item_embedding')(movie_id_input)



# Reshape the embedding layers

user_vector = Reshape([embedding_size])(user_embedding)

movie_vector = Reshape([embedding_size])(movie_embedding)



# Compute dot-product of reshaped embedding layers as prediction

y = Dot(1, normalize=False)([user_vector, movie_vector])



# Setup model

model = Model(inputs=[user_id_input, movie_id_input], outputs=y)

model.compile(loss='mse', optimizer='adam')





# Fit model

model.fit([train_user_data, train_movie_data],

          df_train['Rating'],

          batch_size=256, 

          epochs=4,

          validation_split=0.1,

          shuffle=True)



# Test model

y_pred = model.predict([test_user_data, test_movie_data])

y_true = df_test['Rating'].values



#  Compute RMSE

rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))
# Setup variables

user_embedding_size = 20

movie_embedding_size = 10





##### Create model

# Set input layers

user_id_input = Input(shape=[1], name='user')

movie_id_input = Input(shape=[1], name='movie')



# Create embedding layers for users and movies

user_embedding = Embedding(output_dim=user_embedding_size, 

                           input_dim=users,

                           input_length=1, 

                           name='user_embedding')(user_id_input)

movie_embedding = Embedding(output_dim=movie_embedding_size, 

                            input_dim=movies,

                            input_length=1, 

                            name='item_embedding')(movie_id_input)



# Reshape the embedding layers

user_vector = Reshape([user_embedding_size])(user_embedding)

movie_vector = Reshape([movie_embedding_size])(movie_embedding)



# Concatenate the reshaped embedding layers

concat = Concatenate()([user_vector, movie_vector])



# Combine with dense layers

dense = Dense(256)(concat)

y = Dense(1)(dense)



# Setup model

model = Model(inputs=[user_id_input, movie_id_input], outputs=y)

model.compile(loss='mse', optimizer='adam')





# Fit model

model.fit([train_user_data, train_movie_data],

          df_train['Rating'],

          batch_size=256, 

          epochs=4,

          validation_split=0.1,

          shuffle=True)



# Test model

y_pred = model.predict([test_user_data, test_movie_data])

y_true = df_test['Rating'].values



#  Compute RMSE

rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import vstack
# Create user- & movie-id mapping

user_id_mapping = {id:i for i, id in enumerate(df['User'].unique())}

movie_id_mapping = {id:i for i, id in enumerate(df['Movie'].unique())}



# Use mapping to get better ids

df['User'] = df['User'].map(user_id_mapping)

df['Movie'] = df['Movie'].map(movie_id_mapping)





##### Combine both datasets to get movies with metadata

# Preprocess metadata

tmp_metadata = movie_metadata.copy()

tmp_metadata.index = tmp_metadata.index.str.lower()



# Preprocess titles

tmp_titles = movie_titles.drop('Year', axis=1).copy()

tmp_titles = tmp_titles.reset_index().set_index('Name')

tmp_titles.index = tmp_titles.index.str.lower()



# Combine titles and metadata

df_id_descriptions = tmp_titles.join(tmp_metadata).dropna().set_index('Id')

df_id_descriptions['overview'] = df_id_descriptions['overview'].str.lower()

del tmp_metadata,tmp_titles





# Filter all ratings with metadata

df_hybrid = df.drop('Date', axis=1).set_index('Movie').join(df_id_descriptions).dropna().drop('overview', axis=1).reset_index().rename({'index':'Movie'}, axis=1)





# Split train- & testset

n = 100000

df_hybrid = df_hybrid.sample(frac=1).reset_index(drop=True)

df_hybrid_train = df_hybrid[:1500000]

df_hybrid_test = df_hybrid[-n:]





# Create tf-idf matrix for text comparison

tfidf = TfidfVectorizer(stop_words='english')

tfidf_hybrid = tfidf.fit_transform(df_id_descriptions['overview'])





# Get mapping from movie-ids to indices in tfidf-matrix

mapping = {id:i for i, id in enumerate(df_id_descriptions.index)}



train_tfidf = []

# Iterate over all movie-ids and save the tfidf-vector

for id in df_hybrid_train['Movie'].values:

    index = mapping[id]

    train_tfidf.append(tfidf_hybrid[index])

    

test_tfidf = []

# Iterate over all movie-ids and save the tfidf-vector

for id in df_hybrid_test['Movie'].values:

    index = mapping[id]

    test_tfidf.append(tfidf_hybrid[index])





# Stack the sparse matrices

train_tfidf = vstack(train_tfidf)

test_tfidf = vstack(test_tfidf)





##### Setup the network

# Network variables

user_embed = 10

movie_embed = 10





# Create two input layers

user_id_input = Input(shape=[1], name='user')

movie_id_input = Input(shape=[1], name='movie')

tfidf_input = Input(shape=[24144], name='tfidf', sparse=True)



# Create separate embeddings for users and movies

user_embedding = Embedding(output_dim=user_embed,

                           input_dim=len(user_id_mapping),

                           input_length=1,

                           name='user_embedding')(user_id_input)

movie_embedding = Embedding(output_dim=movie_embed,

                            input_dim=len(movie_id_mapping),

                            input_length=1,

                            name='movie_embedding')(movie_id_input)



# Dimensionality reduction with Dense layers

tfidf_vectors = Dense(128, activation='relu')(tfidf_input)

tfidf_vectors = Dense(32, activation='relu')(tfidf_vectors)



# Reshape both embedding layers

user_vectors = Reshape([user_embed])(user_embedding)

movie_vectors = Reshape([movie_embed])(movie_embedding)



# Concatenate all layers into one vector

both = Concatenate()([user_vectors, movie_vectors, tfidf_vectors])



# Add dense layers for combinations and scalar output

dense = Dense(512, activation='relu')(both)

dense = Dropout(0.2)(dense)

output = Dense(1)(dense)





# Create and compile model

model = Model(inputs=[user_id_input, movie_id_input, tfidf_input], outputs=output)

model.compile(loss='mse', optimizer='adam')





# Train and test the network

model.fit([df_hybrid_train['User'], df_hybrid_train['Movie'], train_tfidf],

          df_hybrid_train['Rating'],

          batch_size=1024, 

          epochs=4,

          validation_split=0.1,

          shuffle=True)



y_pred = model.predict([df_hybrid_test['User'], df_hybrid_test['Movie'], test_tfidf])

y_true = df_hybrid_test['Rating'].values



rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))