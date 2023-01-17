import os

import pprint

import operator

import numpy as np

import pandas as pd

from os import listdir

from sklearn.metrics.pairwise import pairwise_distances 
#Information about the data

info = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.info", sep=" ", header=None)

info.columns=['Counts', 'Type']

info
#Types of genres

genre = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.genre', sep="|", encoding='latin-1', header=None)

genre.drop(genre.columns[1], axis=1, inplace=True)

genre.columns = ['Genres']

genre_list = list(genre['Genres'])

genre_list
#Types of occupations

occupation = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.occupation', sep="|", encoding='latin-1', header=None)

occupation.columns = ['Occupations']

occupation_list = list(occupation['Occupations'])

occupation_list
#Load the Ratings data

data = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', sep="\t", header=None)

data.columns = ['user id', 'movie id', 'rating', 'timestamp']

data.head()
#Load the Movies data

item = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep="|", encoding='latin-1', header=None)

item.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 

                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 

                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

item.head()
#Load the User data

user = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', sep="|", encoding='latin-1', header=None)

user.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

user.head()
#merge the 'data' table with 'user' table

data_user = pd.merge(data[['user id', 'movie id', 'rating']], user[['user id', 'occupation']], on='user id')

data_user.drop(columns = ['user id'], inplace=True)



#merge the 'Data_User' dataframe with 'Item' dataframe to get each rating, occupation of user and movie title

data_user_item = pd.merge(data_user[['movie id', 'rating', 'occupation']], item[['movie id', 'movie title']], on='movie id')

data_user_item.drop(columns = ['movie id'], inplace=True)



#group the data by occupation and movie title and sort with decreasing average ratings

data_user_item_sorted = data_user_item.groupby(['occupation', 'movie title'], as_index=False)['rating'].mean().sort_values('rating', ascending=False)



#group data by occupation, movie title and select top 3 movies for each occupation

top_3_occ = data_user_item_sorted.groupby(['occupation']).head(3).sort_values(['occupation', 'movie title'], ascending=[True, True]).reset_index()

top_3_occ.drop(['index'], axis=1, inplace=True)



top_3_occ
#merge the 'data' table with the 'item' table

data_item = pd.merge(data[['user id', 'movie id', 'rating']], item, on='movie id')

data_item.drop(columns = ['user id', 'movie id', 'release date', 'video release date', 'IMDb URL'], inplace=True)



#For each genre get the top 3 movies by average rating

top_3_genre = pd.DataFrame()

for gen in genre_list:

    g_r = data_item[data_item[gen] == 1]

    new_gen = pd.DataFrame(g_r.groupby(['movie title'], as_index=False)['rating'].mean().sort_values(['rating', 'movie title'], ascending=[False, True]).head(3))

    new_gen.insert(0, 'genre', gen)

    top_3_genre = top_3_genre.append(new_gen, ignore_index=True)



top_3_genre
#merge the 'data' table with 'user' table

data_user_og = pd.merge(data[['user id', 'movie id', 'rating']], user[['user id', 'occupation']], on='user id')

data_user_og.drop(columns = ['user id'], inplace=True)



#merge the 'data_user_og' dataframe with 'Item' dataframe to get each rating, occupation of user and movie title

data_user_item_og = pd.merge(data_user_og[['movie id', 'rating', 'occupation']], item, on='movie id')

data_user_item_og.drop(columns = ['movie id'], inplace=True)



#find the top 3 movies in each genre for each occupation

top_3_occ_genre = pd.DataFrame()

for occ in list(occupation['Occupations']):

    occ_table = data_user_item_og[data_user_item_og['occupation']==occ]

    for gen in genre_list:

        g_o_r = occ_table[occ_table[gen] == 1]

        new_occ_gen = pd.DataFrame(g_o_r.groupby(['movie title'], as_index=False)['rating'].mean().sort_values(['rating', 'movie title'], ascending=[False, True]).head(3))

        new_occ_gen.insert(0, 'genre', gen)

        new_occ_gen.insert(0, 'occupation', occ)

        top_3_occ_genre = top_3_occ_genre.append(new_occ_gen, ignore_index=True)

        

top_3_occ_genre
#create a column of age group in the users dataframe

bins= [0,6,12,18,30,50,200]

labels = ['<=6','<=12','<=18','<=30','<=50', '50+']

user['age group'] = pd.cut(user['age'], bins=bins, labels=labels, right=True)



#merge the 'data' table with the 'user' table

data_user_age = pd.merge(data[['user id', 'movie id', 'rating']], user[['user id', 'age group']], on='user id')

data_user_age.drop(columns = ['user id'], inplace=True)



#merge the 'data_user_age' table with the 'item' table to get each rating, age group of user and movie title

data_user_item_age = pd.merge(data_user_age[['movie id', 'rating', 'age group']], item[['movie id', 'movie title']], on='movie id')

data_user_item_age.drop(columns = ['movie id'], inplace=True)

data_user_item_age['age group'] = data_user_item_age['age group'].astype('category')



#group the data by age group and movie title and sort with decreasing average ratings

data_user_item_age_sorted = data_user_item_age.groupby(['age group', 'movie title'], as_index=False)['rating'].mean().sort_values('rating', ascending=False)



#group data by occupation, movie title and select top 3 movies for each occupation

top_3_age = data_user_item_age_sorted.groupby(['age group']).head(3).sort_values(['age group', 'movie title'], ascending=[True, True]).reset_index()

top_3_age.drop(['index'], axis=1, inplace=True)



top_3_age
#merge 'data' table with 'item' table and count movies by genre

data_item_gen = pd.merge(data[['user id', 'movie id', 'rating']], item, on='movie id')

data_item_gen['release date 2'] = pd.to_datetime(data_item_gen['release date'])

data_item_gen['release month'] = data_item_gen['release date 2'].dt.month

data_item_gen_summer = data_item_gen[(data_item_gen['release month']>=5) & (data_item_gen['release month']<=7)]



count_gen = {}

for gen in genre_list:

    count_gen[gen] = sum(data_item[gen])



sorted_count_gen = sorted(count_gen.items(), key=operator.itemgetter(1), reverse=True)

sorted_count_gen[:3]
top_3_genre_summer = pd.DataFrame(columns = ['genre', 'average rating'])

for gen in genre_list:

    genre_this = data_item_gen_summer[data_item_gen_summer[gen] == 1]

    row = [gen, genre_this['rating'].mean()]

    top_3_genre_summer.loc[len(top_3_genre_summer)] = row

    

top_3_genre_summer_res = top_3_genre_summer.sort_values('average rating', ascending=False).head(3)

# top_3_genre_summer_res.to_csv('Top3GenresSummer.csv', index=False, sep=',')

top_3_genre_summer_res
#merge the 'data' table with the 'item' table

data_item_gen_co = pd.merge(data[['user id', 'movie id', 'rating']], item, on='movie id')



#for each genre, calculate the top 2 co-occurring genres and store results in a dictionary

top2gens = {}

for gen1 in genre_list:

    t = {}

    for gen2 in genre_list:

        if gen1 != gen2:

            t[gen2] = data_item_gen_co[(data_item_gen_co[gen1]==1) & (data_item_gen_co[gen2]==1)].shape[0]

    sorted_t = sorted(t.items(), key=operator.itemgetter(1), reverse=True)

    top2gens[gen1] = sorted_t[:2]



top2gens
#create a user-rating matrix

data_matrix = np.zeros((info['Counts'][0], info['Counts'][1]))

for line in data.itertuples():

    data_matrix[line[1]-1, line[2]-1] = line[3]



#calculate user similarity matrix based on cosine similarity of ratings

user_similarity = pairwise_distances(data_matrix, metric='cosine')



#create dictionary of each user and user with most simiar preferances

sim_user= {}

for i in range(943):

    sim_user[i+1] = [np.argmax(user_similarity[i])+1]



#The dictionary sim_user contains the most similar user to every user in the data.