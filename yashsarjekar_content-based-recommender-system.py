# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from scipy.stats import pearsonr
import missingno
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
movies_df = pd.read_csv('/kaggle/input/movies-recomendation-system/movies.csv')
ratings_df = pd.read_csv('/kaggle/input/movies-recomendation-system/ratings.csv')
movies_df.head()
ratings_df.head()
shape = []
shape.append(movies_df.shape)
shape.append(ratings_df.shape)
shape
missingno.matrix(movies_df)
missingno.matrix(ratings_df)
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()
copy_movies_df = movies_df.copy()
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        copy_movies_df.at[index,genre] = 1
copy_movies_df = copy_movies_df.fillna(0)
copy_movies_df.head()
ratings_df.drop('timestamp',axis=1,inplace=True)
ratings_df.head()
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
user_movie_id = pd.merge(inputMovies,movies_df,on='title')
user_movie_id
user_movie_id.drop('genres',axis=1,inplace=True)
user_movie_id.drop('year',axis=1,inplace=True)
user_movie_id
user_movie_id.sort_values(by=['movieId'],inplace=True)
user_movie_id
genres_movies_df = copy_movies_df[copy_movies_df['movieId'].isin(user_movie_id['movieId'].tolist())]
genres_movies_df
genres_five_movies = genres_movies_df.drop('movieId',axis=1).drop('title',axis=1).drop('genres',axis=1).drop('year',axis=1)
genres_five_movies
genres_five_movies = genres_five_movies.reset_index(drop=True)
genres_five_movies
user_profile = genres_five_movies.transpose().dot(user_movie_id['rating'])
user_profile
copy_movies_df.drop(copy_movies_df.index[copy_movies_df.movieId == 1],axis=0,inplace=True)
copy_movies_df.drop(copy_movies_df.index[copy_movies_df.movieId == 2],axis=0,inplace=True)
copy_movies_df
copy_movies_df.drop(copy_movies_df.index[copy_movies_df.movieId == 296],axis=0,inplace=True)
copy_movies_df.drop(copy_movies_df.index[copy_movies_df.movieId == 1274],axis=0,inplace=True)
copy_movies_df.drop(copy_movies_df.index[copy_movies_df.movieId == 1968],axis=0,inplace=True)
copy_movies_df
genres = copy_movies_df.set_index(copy_movies_df['movieId'])
genres
genres
genres.drop('title',axis=1,inplace=True)
genres.drop('genres',axis=1,inplace=True)
genres.drop('year',axis=1,inplace=True)
genres
user_profile.shape
recommender_table_df = ((genres*user_profile).sum(axis=1,))/(user_profile.sum())
recommender_table_df.head()
recommender_table_df = recommender_table_df.sort_values(ascending=False)
recommender_table_df.head(20)
movies_df.loc[movies_df['movieId'].isin(recommender_table_df.head(20).keys())]
coll_movies_df = movies_df.copy()
coll_movies_df.head()
coll_ratings_df = ratings_df.copy()
coll_ratings_df.head()
coll_movies_df.drop('genres',axis=1,inplace=True)
coll_movies_df.head()
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':4},
            {'title':'Jumanji', 'rating':3},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
inputMovies = pd.merge(coll_movies_df,inputMovies,on='title')
inputMovies.drop('year',axis=1,inplace=True)
inputMovies
users_watched_movies = coll_ratings_df[coll_ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
users_watched_movies
users_watched_movies_group = users_watched_movies.groupby(['userId'])
users_watched_movies_group
users_watched_movies_group.get_group(1130)
users_watched_movies_group_sort = sorted(users_watched_movies_group,  key=lambda x: len(x[1]), reverse=True)
users_watched_movies_group_sort[0:5]
users_watched_movies_group_sort =users_watched_movies_group_sort[0:100]
pearson_correlation_dict = {}
for name, group in users_watched_movies_group_sort:
    group = group.sort_values('movieId')
    inputMovies = inputMovies.sort_values('movieId')
    ## get the movies common 
    tmp = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #print(tmp)
    tmp_rating = tmp['rating'].tolist()
    grp_rating = group['rating'].tolist()
    ## Now Calculate the pearson Correlation between two users. 
    corr,_ = pearsonr(tmp_rating,grp_rating)
    if corr != 0:
        pearson_correlation_dict[name] = corr
    else:
        pearson_correlation_dict[name] = 0
pearson_correlation_dict.items()
pearsonDf  = pd.DataFrame.from_dict(pearson_correlation_dict,orient='index')
pearsonDf.columns = ['similarityIndex']
pearsonDf['userId'] = pearsonDf.index
pearsonDf.index = range(len(pearsonDf))
pearsonDf.head()
top_users = pearsonDf.sort_values('similarityIndex',ascending=False)[0:50]
top_users.head()
topUsersRating=top_users.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()
topUsersRating['weightRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
temptopuserrating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightRating']]
temptopuserrating.columns = ['sum_similarityIndex','sum_weightedRating']
temptopuserrating.head()
recommedation = pd.DataFrame()
recommedation['weights_actual'] = temptopuserrating['sum_similarityIndex']/temptopuserrating['sum_weightedRating']
recommedation['movieId'] = temptopuserrating.index
recommedation.head()
recommedation = recommedation.sort_values(by='weights_actual',ascending=False)
recommedation.head()
recommedation
coll_movies_df.loc[coll_movies_df['movieId'].isin(recommedation.head(10)['movieId'].tolist())]
