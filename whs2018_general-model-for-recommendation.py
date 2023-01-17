!pip install deepctr
# To store the data

import pandas as pd



# To do linear algebra

import numpy as np



# To create plots

import matplotlib.pyplot as plt



# # To create interactive plots

# from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs

# import plotly as py

# import plotly.graph_objs as go

# # init_notebook_mode(connected=True)

# To operator files

import os

# To shift lists

from collections import deque



# To compute similarities between vectors

from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# To use recommender systems

import surprise as sp

from surprise.model_selection import cross_validate



# To create deep learning models

from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout

from keras.models import Model



# To create sparse matrices

from scipy.sparse import coo_matrix



# To light fm

from lightfm import LightFM

from lightfm.evaluation import precision_at_k



# To deepctr

from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

from deepctr.models import DeepFM, xDeepFM, DCN, DIN, DSIN, DIEN



# To stack sparse matrices

from scipy.sparse import vstack
# 加载netflix-prize-data数据集

os.listdir('../input/netflix-prize-data/')

# qualifying.txt:要提交的预测文件

# MovieID1:

# CustomerID11,Date11

# CustomerID12,Date12

# -> 

# MovieID1:

# Rating11

# Rating12



# probe.txt: 和qualifying.txt文件类似，与之不同的是没有Date列



# movie_titles.txt : 电影信息，数据格式为MovieId, YearOfRelease, Title

# combined_data_1/2/3/4.txt ： 训练集， 数据格式为CustomerID(user), Rating, Date
movie_netflix = pd.read_csv('../input/netflix-prize-data/movie_titles.csv', 

                           encoding = 'ISO-8859-1', 

                           header = None, 

                           names = ['Id', 'Year', 'Name']).set_index('Id')



print('Shape Movie-Titles:\t{} \n Contains {} items'.format(movie_netflix.shape, movie_netflix.shape[0]))

movie_netflix.sample(5)
# 加载the-movies-dataset数据集

# os.listdir('../input/the-movies-dataset')

# movies_metadata.csv: 电影元文件，每个电影共计24个特征

# keywords.csv: id-keyword，每个电影对应一个关键词

# credits.csv: id-cast-crew，每个电影对应摄制组和演员信息

# links.csv: id-imdbid-tmdbid，不同电影平台对同一部电影的不用标识

# ratings_small.csv : 评分数据，userId-movieId-rating-timestamp
# low_memory=False关键词

# low_memory=False 参数设置后，pandas会一次性读取csv中的所有数据，然后对字段的数据类型进行唯一的一次猜测。这样就不会导致同一字段的Mixed types问题了。

# 但是这种方式真的非常不好，一旦csv文件过大，就会内存溢出；

# movie_metadata = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)[['original_title', 'id', 'release_date', 'vote_count']].set_index('id')

# # 移除投票次数小于10的样本

# movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)



# print('Shape Movie-Metadata:\t{}\n Contains {} items'.format(movie_metadata.shape, movie_metadata.shape[0]))

# movie_metadata.sample(5)
# 加载movielens20m数据集

# os.listdir('../input/movielens-20m-dataset/')

# tag.csv: userId-movieId-tag-timestamp

# rating.csv: userId-movieId-rating-timestamp

# movie.csv: movieId-title-genres

# link.csv: moiveId-imdbId-tmbdId

# genome_scores.csv: movieId-tagId-relevance

# genome_tags.csv: tagId-tag
# movie_movielens = pd.read_csv('../input/movielens-20m-dataset/movie.csv').set_index('movieId')

# print('Shape MovieLens-movice:\t{}\n Contains {} items'.format(movie_movielens.shape, movie_movielens.shape[0]))

# movie_movielens.head(5)
# Load single data-file 

# combined_data_1 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# combined_data_2 = pd.read_csv('../input/netflix-prize-data/combined_data_2.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# combined_data_3 = pd.read_csv('../input/netflix-prize-data/combined_data_3.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# combined_data_4 = pd.read_csv('../input/netflix-prize-data/combined_data_4.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# df_raw = pd.cocat([combined_data_1, combined_data_2, combined_data_3, combined_data_4], axis=0).reset_index()

# 鉴于netflix-prize-data中存在

df_raw = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['userId', 'rating', 'Date'], usecols=[0, 1, 2])

print('Shape Raw Data:\t{}'.format(df_raw.shape))



# Find empty rows to slice dataframe for each movie

# 编码思路是先找出缺失值的索引，然后遍历过滤掉索引值

tmp_movies = df_raw[df_raw['rating'].isna()]['userId'].reset_index()

movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values] # drop ':'



# Shift the movie_indices by one to get start and endpoints of all movies

shifted_movie_indices = deque(movie_indices)

shifted_movie_indices.rotate(-1)  # the first element turn to the last element.





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

    tmp_df['movieId'] = movie_id

    

    # Append dataframe to list

    user_data.append(tmp_df)



# Combine all dataframes

netflix_prize_User = pd.concat(user_data)

del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

print('Shape User-Ratings:\t{}'.format(netflix_prize_User.shape))

netflix_prize_User.sample(5)
# movie_dataset_User = pd.read_csv('../input/the-movies-dataset/ratings.csv', low_memory=False)

# print('Shape User-Ratings:\t{}'.format(movie_dataset_User.shape))

# movie_dataset_User.head(5)
# movielens_movie_User = pd.read_csv('../input/movielens-20m-dataset/rating.csv')

# print('Shape MovieLens-movice:\t{}'.format(movielens_movie_User.shape))

# movielens_movie_User.head(5)
def filter_user_item(user_item_rating, min_nb_item_ratings=300, min_nb_user_ratings=200):

    filter_items = (user_item_rating['movieId'].value_counts() > min_nb_item_ratings)

    filter_items = filter_items[filter_items].index.tolist()

    

    filter_users = (user_item_rating['userId'].value_counts() > min_nb_user_ratings)

    filter_users = filter_users[filter_users].index.tolist()

    filter_ret = user_item_rating[(user_item_rating['movieId'].isin(filter_items)) & (user_item_rating['userId'].isin(filter_users))]

    print('Shape User-Ratings unfiltered:\t{}'.format(user_item_rating.shape))

    print('Shape User-Ratings filtered:\t{}'.format(filter_ret.shape))

    return filter_ret
# netflix_prize_User

filtered_netflix_prize_User = filter_user_item(netflix_prize_User)

# filtered_movie_dataset_User = filter_user_item(movie_dataset_User)

# filtered_movielens_movie_User = filter_user_item(movielens_movie_User)
del netflix_prize_User#, movie_dataset_User, movielens_movie_User
def get_train_test(filtered_user_item, test_size=0.5):

    X_train, X_test, _, _ = train_test_split(filtered_user_item.reset_index(), filtered_user_item['movieId'].values, test_size=test_size, random_state=2020, stratify=filtered_user_item['movieId'].values)

    return X_train, X_test
# train_data1, test_data1 = get_train_test(filtered_movie_dataset_User)

# movieId1 = train_data1.movieId

# userId1 = train_data1.userId

# train_data2, test_data2 = get_train_test(filtered_movielens_movie_User)

# movieId2 = train_data2.movieId

# userId2 = train_data2.userId

train_data3, test_data3 = get_train_test(filtered_netflix_prize_User)

movieId3 = train_data3.movieId

userId3 = train_data3.userId

# del filtered_movie_dataset_User, filtered_movielens_movie_User, filtered_netflix_prize_User

# del filtered_netflix_prize_User
def get_user_item_rating_mat(data):

    return data.pivot_table(index='userId', columns='movieId', values='rating')
# train_data1 = get_user_item_rating_mat(train_data1)

# train_data2 = get_user_item_rating_mat(train_data2)

matrix_train_data3 = get_user_item_rating_mat(train_data3)

# train_data1.sample(4), train_data2.sample(4), train_data3.sample(4)

matrix_train_data3.head(5)
# train_data1.to_csv('train_data1.csv', index=False, header=None)

# train_data2.to_csv('train_data2.csv', index=False, header=None)

# train_data3.to_csv('train_data3.csv', index=False, header=None)
# del train_data1, train_data2, train_data3

# del train_data3
def mean_rating(train, test):

    # 0：表示沿着每一列或行标签/索引值向下执行方法

    # 1：表示沿着每一行或列标签/索引值向右执行方法

    ratings_mean = train.mean(axis=0).rename('rating_mean')

    df_pred = test.set_index('movieId').join(ratings_mean)[['rating', 'rating_mean']]

#     df_pred.fillna(df_pred.mean(), inplace=True)

    rmse = np.sqrt(mean_squared_error(y_true=df_pred['rating'], y_pred=df_pred['rating_mean']))

    print("mean rating's rmse is {}".format(rmse))

    return rmse
# train_data3 = pd.read_csv('./train_data3.csv',header=None)

# train_data3.head(5)
# train_data3.index = userId3

# train_data3.columns = movieId3 
# train_data3 = pd.read_csv('./train_data3.csv', header=None, index_col=userId3.values, names=movieId3.values)

# mean_rating_data1 = mean_rating(train_data3, test_data3)

# del train_data3
# train_data1 = pd.read_csv('./train_data1.csv')

# mean_rating_data1 = mean_rating(train_data1, test_data1)

# del train_data1

# train_data2 = pd.read_csv('./train_data2.csv')

# mean_rating_data2 = mean_rating(train_data2, test_data2)

# del train_data2

# train_data3 = pd.read_csv('./train_data3.csv')

# mean_rating_data3 = mean_rating(train_data3, test_data3)

# del train_data3

mean_rating_data3 = mean_rating(matrix_train_data3, test_data3)
def weighted_mean_rating(train, test, m=1000):

    C = train.stack().mean()  # 一个浮点数

    """

    数据格式如下：

    userId1:

    movieId11, rating

    movieId12, rating

    userId2:

    movieId21, rating

    movieId22, rating

    """

    R = train.mean(axis=0).values # movie个数的一个array，每个值为rating的平均值

    v = train.count().values # movie个数的一个array，每个值为user的个数

    weighted_score = (v/ (v+m) *R) + (m/ (v+m) *C)

    df_prediction = test.set_index('movieId').join(pd.DataFrame(weighted_score, index=train.columns, columns=['prediction']))[['rating', 'prediction']]

    y_true = df_prediction['rating']

    y_pred = df_prediction['prediction']

    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

    print('weighted mean rating"s rmse is {}'.format(rmse))

    return rmse
weighted_mean_rating_data3 = weighted_mean_rating(matrix_train_data3, test_data3, 50)
def cosine_u2u_similarity(train, test, n_recommendation=100):

    train_imputed = train.T.fillna(train.mean(axis=1)).T  # 利用均值进行填充NaN

    similarity = cosine_similarity(train_imputed.values)  # 计算用户之间的余弦相似度

    similarity -= np.eye(similarity.shape[0]) # 减去自身相似度

    

    prediction = []

    userId_idx_mapping = {userId:idx for idx, userId in enumerate(train_imputed.index)}

    for userId in test.userId.unique():

        similarity_user_index = np.argsort(similarity[userId_idx_mapping[userId]])[::-1]

        similarity_user_score = np.sort(similarity[userId_idx_mapping[userId]])[::-1]

        for movieId in test[test.userId == userId].movieId.values:

            

            score = (train_imputed.iloc[similarity_user_index[:n_recommendation]][movieId] * similarity_user_score[:n_recommendation]).values.sum() / similarity_user_score[:n_recommendation].sum()

            prediction.append([userId, movieId, score])

    

    # Create prediction DataFrame

    df_pred = pd.DataFrame(prediction, columns=['userId', 'movieId', 'prediction']).set_index(['userId', 'movieId'])

    df_pred = test.set_index(['userId', 'movieId']).join(df_pred)





    # Get labels and predictions

    y_true = df_pred['rating'].values

    y_pred = df_pred['prediction'].values



    # Compute RMSE

    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

    print("consine_u2u_similarity's rmse is {}".format(rmse))

    return rmse
cosine_u2u_similarity_data3 = cosine_u2u_similarity(matrix_train_data3, test_data3, n_recommendation=100)
def matrix_factorization_dot(train, test, embedding_size=50):

    userId_idx_mapping = {userId:idx for idx, userId in enumerate(train.userId.unique())}

    movieId_idx_mapping = {movieId:idx for idx, movieId in enumerate(train.movieId.unique())}

    # 和reset_index函数一样，为了方便NN模型的输入（主要体现在Batch的获取上）

    train_user_data = train.userId.map(userId_idx_mapping)

    train_movie_data = train.movieId.map(movieId_idx_mapping)

    

    test_user_data = test.userId.map(userId_idx_mapping)

    test_movie_data = test.movieId.map(movieId_idx_mapping)

    

    nb_users = len(userId_idx_mapping)

    nb_movies = len(movieId_idx_mapping)

    

    

    # 创建模型

    # 定义输入，维度

    userId_input = Input(shape=[1], name='user')

    movieId_input = Input(shape=[1], name='movie')

    # 创建embedding层

    user_embedding = Embedding(

        output_dim=embedding_size,

        input_dim=nb_users,

        input_length=1,

        name='user_embedding'

    )(userId_input)

    

    movie_embedding = Embedding(

        output_dim=embedding_size,

        input_dim=nb_movies,

        input_length=1,

        name='movie_embedding'

    )(movieId_input)

    # Reshape the embedding layers

    user_vector = Reshape([embedding_size])(user_embedding)

    movie_vector = Reshape([embedding_size])(movie_embedding)



    # Compute dot-product of reshaped embedding layers as prediction

    y = Dot(1, normalize=False)([user_vector, movie_vector])



    # Setup model

    model = Model(inputs=[userId_input, movieId_input], outputs=y)

    model.compile(loss='mse', optimizer='adam')





    # Fit model

    model.fit([train_user_data, train_movie_data],

              train.rating,

              batch_size=256, 

              epochs=10,

              validation_split=0.4,

              shuffle=True)



    # Test model

    y_pred = model.predict([test_user_data, test_movie_data])

    y_true = test.rating.values



    #  Compute RMSE

    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

    print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))

    return rmse
matrix_factorization_dot_train3 = matrix_factorization_dot(train_data3, test_data3)
def matrix_factorization_dnn(train, test, nb_user_embedding=20, nb_movie_embedding=40):

    userId_idx_mapping = {userId:idx for idx, userId in enumerate(train.userId.unique())}

    movieId_idx_mapping = {movieId:idx for idx, movieId in enumerate(train.movieId.unique())}

    

    # Create correctly mapped train- & testset

    train_user_data = train.userId.map(userId_idx_mapping)

    train_movie_data = train.movieId.map(movieId_idx_mapping)



    test_user_data = test.userId.map(userId_idx_mapping)

    test_movie_data = test.movieId.map(movieId_idx_mapping)

    

    nb_users = len(userId_idx_mapping)

    nb_movies = len(movieId_idx_mapping)

    ##### Create model

    # Set input layers

    userId_input = Input(shape=[1], name='user')

    movieId_input = Input(shape=[1], name='movie')



  

    

    # Create embedding layers for users and movies

    user_embedding = Embedding(output_dim=nb_user_embedding, 

                               input_dim=nb_users,

                               input_length=1, 

                               name='user_embedding')(userId_input)

    movie_embedding = Embedding(output_dim=nb_movie_embedding, 

                                input_dim=nb_movies,

                                input_length=1, 

                                name='item_embedding')(movieId_input)



    # Reshape the embedding layers

    user_vector = Reshape([nb_user_embedding])(user_embedding)

    movie_vector = Reshape([nb_movie_embedding])(movie_embedding)



    # Concatenate the reshaped embedding layers

    concat = Concatenate()([user_vector, movie_vector])



    # Combine with dense layers

    dense = Dense(256)(concat)

    y = Dense(1)(dense)



    # Setup model

    model = Model(inputs=[userId_input, movieId_input], outputs=y)

    model.compile(loss='mse', optimizer='adam')





    # Fit model

    model.fit([train_user_data, train_movie_data],

              train.rating,

              batch_size=256, 

              epochs=5,

              validation_split=0.5,

              shuffle=True)



    # Test model

    y_pred = model.predict([test_user_data, test_movie_data])

    y_true = test.rating.values



    #  Compute RMSE

    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

    print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))

    return rmse
matrix_factorization_dnn_train3 = matrix_factorization_dnn(train_data3, test_data3)
def surprise_library(data):

    # Load dataset into surprise specific data-structure

    sampled_data = sp.Dataset.load_from_df(data[['userId', 'movieId', 'rating']].sample(20000), sp.Reader())



    benchmark = []

    # Iterate over all algorithms

    for algorithm in [sp.SVD(), sp.SVDpp(), sp.SlopeOne(), sp.NMF(), sp.NormalPredictor(), sp.KNNBaseline(), sp.KNNBasic(), sp.KNNWithMeans(), sp.KNNWithZScore(), sp.BaselineOnly(), sp.CoClustering()]:

        # Perform cross validation

        results = cross_validate(algorithm, sampled_data, measures=['RMSE', 'MAE'], cv=3, verbose=False)



        # Get results & append algorithm name

        tmp = pd.DataFrame.from_dict(results).mean(axis=0)

        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))



        # Store data

        benchmark.append(tmp)

    return benchmark
surprise_train3 = surprise_library(filtered_netflix_prize_User)
surprise_train3 = pd.DataFrame(surprise_train3).set_index('Algorithm')
# surprise_train3['test_rmse'].tolist()
def lightfm_library(train, test):

    # Create user- & movie-id mapping

    user_id_mapping = {id:i for i, id in enumerate(train['userId'].unique())}

    movie_id_mapping = {id:i for i, id in enumerate(train['movieId'].unique())}

    

    # Create correctly mapped train- & testset

    train_user_data = train['userId'].map(user_id_mapping)

    train_movie_data = train['movieId'].map(movie_id_mapping)



    test_user_data = test['userId'].map(user_id_mapping)

    test_movie_data = test['movieId'].map(movie_id_mapping)





    # Create sparse matrix from ratings

    shape = (len(user_id_mapping), len(movie_id_mapping))

    train_matrix = coo_matrix((train['rating'].values, (train_user_data.astype(int), train_movie_data.astype(int))), shape=shape)

    test_matrix = coo_matrix((test['rating'].values, (test_user_data.astype(int), test_movie_data.astype(int))), shape=shape)





    # Instantiate and train the model

    model = LightFM(loss='warp', no_components=20)

    model.fit(train_matrix, epochs=10, num_threads=2)





    # Evaluate the trained model

    k = 20

    precision_score = precision_at_k(model, test_matrix, k=k).mean()

#     print('Train precision at k={}:\t{:.4f}'.format(k, precision_at_k(model, train_matrix, k=k).mean()))

    print('Test precision at k={}:\t\t{:.4f}'.format(k, precision_score))

    return precision_score
lightfm_train3 = lightfm_library(train_data3, test_data3)
## DeepFM

def deepfm_algo(data):



    sparse_features = ["movieId", "userId"]

    target = ['rating']

    for feat in sparse_features:

            lbe = LabelEncoder()

            data[feat] = lbe.fit_transform(data[feat])

    

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)

                              for feat in sparse_features]

    

    linear_feature_columns = fixlen_feature_columns

    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    

    train, test = train_test_split(data, test_size=0.5)

    train_model_input = {name:train[name].values for name in feature_names}

    test_model_input = {name:test[name].values for name in feature_names}



    # 4.Define Model,train,predict and evaluate

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

    model.compile("adam", "mse", metrics=['mse'], )

    

    history = model.fit(train_model_input, train[target].values,

                        batch_size=256, epochs=5, verbose=2, validation_split=0.5, )

    pred_ans = model.predict(test_model_input, batch_size=256)

    rmse = np.sqrt(mean_squared_error(test[target].values, pred_ans))

    print("test RMSE", rmse)

    return rmse
deepfm_algor_train3 = deepfm_algo(filtered_netflix_prize_User)
ret_rmse = [mean_rating_data3, weighted_mean_rating_data3, cosine_u2u_similarity_data3, matrix_factorization_dot_train3, matrix_factorization_dnn_train3, lightfm_train3, deepfm_algor_train3] + surprise_train3['test_rmse'].tolist() 

ret_rmse_name = ['mean_rating', 'weighted', 'cosine_u2u_similarity', 'mf_dot', 'mf_dnn', 'lightfm', 'deepfm'] + surprise_train3.index.tolist()

figure, ax = plt.subplots(figsize=(16,4))

print(ret_rmse)

plt.bar(range(len(ret_rmse)), ret_rmse, tick_label=ret_rmse_name)

for tick in ax.get_xticklabels():

    tick.set_rotation(90)

plt.title('Different RMSE in Dataset by RS algorithm')

plt.show()