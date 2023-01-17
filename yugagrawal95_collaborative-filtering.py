#import all required packages..



import pandas as pd

import numpy as np

import math

from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats.stats import pearsonr

from sklearn.metrics import pairwise_distances
#import Rating, Movie and Tags Data basically Movie and Tag are just mapping for Movie and Tag

# we are using only Rating Frame as we need to deal with movie and rating with user...



Ratings=pd.read_csv("../input/ratings.csv",encoding="ISO-8859-1")

Movies=pd.read_csv("../input/movies.csv",encoding="ISO-8859-1")

Tags=pd.read_csv("../input/tags.csv",encoding="ISO-8859-1")
Ratings.head()
#Get movie rating count so it will be easy at end to rank movie when 2 movie get same rating....

#this is optional if you want can do it else skip this step.



#1. take groupby with respect to movie and get mean avg of each movie. 

movie_rating_count = Ratings.groupby('movieId')['rating'].count()



#2. convert into dataframe for better operation work.

movie_rating_count = pd.DataFrame(movie_rating_count)



#3. change column name.

movie_rating_count.columns = ['rating_count']



#4. create new column movieIUd

movie_rating_count['movieId'] = movie_rating_count.index

movie_rating_count.reset_index(drop=True)



#merging to Rating dataframe so we have collectively all information together..

Ratings = Ratings.merge(movie_rating_count,right_on='movieId',left_on='movieId')
Ratings.head()

Movies.head()
Tags.head()
#create a pivot matrix for user and movie based on rating value..

RatingMat = Ratings.pivot_table(index=['userId'],columns=['movieId'],values=['rating'],fill_value=0)



#keep a copy of original matrix for future refrence..

Original_RatingMat = RatingMat.copy()

RatingMat.head()
#mean normalize to adjust rating on same scale for all user i.e. mean 0 across rows.

# this is because for user1 best mean 5 but user2 best means 4.



RatingMat = RatingMat.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)



#drop multilevel column for better usabiltity...

RatingMat.columns = RatingMat.columns.droplevel()

RatingMat.head(5)


#user wise similarity with cosine...

user_similarity = cosine_similarity(RatingMat)

user_sim_df = pd.DataFrame(user_similarity,index=RatingMat.index,columns=RatingMat.index)

user_sim_df.head(5)



#we can do it using pearson correlation as well there is no hard and fast rule.

# generally user-user approach we use pearson correlation and item-item we use cosine similarity





#user wise similarity with pearson coorelation...

# user_similarity = 1-pairwise_distances(RatingMat, metric="correlation")

# user_sim_df = pd.DataFrame(user_similarity,index=RatingMat.index,columns=RatingMat.index)

# user_sim_df.head(5)

# 1. select any user randomly for example we have selected userid 359 

#Get a list of movie user already watched so we can remove from our recommendation list..



curr_user_rated_movie = Ratings[(Ratings.userId == 359) & (Ratings.rating != 0)]['movieId']

curr_user_rated_movie = pd.DataFrame(curr_user_rated_movie,columns=['movieId'])
# put similarity of current user i.e. 359 in a dataframe because later we need for weighted average..



curr_user_similarity = pd.DataFrame(user_sim_df.loc[359])



#just changing column name as similarity for better readability.

curr_user_similarity.rename(columns={359:'similarity'},inplace=True)

curr_user_similarity.head()
# Remove movieId which user already watched from our rating matrix..

# above we have calculated which movie user has watched.

# subtract or negeate from rating matrix we get still not watched movie..



RatingMatTranspose = RatingMat.T

RatingMatTranspose = RatingMatTranspose.loc[~RatingMatTranspose.index.isin(curr_user_rated_movie.movieId.tolist())]
RatingMatTranspose.head()
#when we have taken pivot table it created multi level column name , so drop for better understanding..



Original_RatingMat.columns = Original_RatingMat.columns.droplevel()
'''

calculate the weighted average a movie per user so later we can show a user k specific - 

avg specific predicition rating for a movie



formulas :- sum(w[1]*user1+w[2]*user2.....w[n]*userN) / w[1]+w[2]....w[n] 

'''           

Weighed_avg = []



for movieId in RatingMatTranspose.index:

    

    '''

     User not watched movie consider as 0

     we need to remove those weight where user has not watched movie in weighted sum.

     In weighted sum in denominator we need to sum all weights

     we will remove weights for those rating is 0.

    '''

    

    user_not_rated = Original_RatingMat[Original_RatingMat[movieId] == 0].index

    

    #calculating weights.

    Total_weight = np.sum(curr_user_similarity.loc[~curr_user_similarity.index.isin(user_not_rated.tolist())]['similarity'])

    

    #appending in weighted_avg list to get final list of weighted avg movie.

    Weighed_avg.append(np.dot(RatingMatTranspose.loc[movieId],curr_user_similarity.similarity) / (Total_weight))
#converting Weighted list in dataframe for better operation works..



Weighed_avg = pd.DataFrame(Weighed_avg,columns=['weighted_avg'])

Weighed_avg.index = RatingMatTranspose.index
#Weighed_avg = Weighed_avg.loc[~Weighed_avg.index.isin(curr_user_rated_movie.movieId.tolist())]



#sort top 10 recommendation movie based on weighted avg here you can append movie count as well if you want.

Weighed_avg.sort_values(by='weighted_avg',ascending=False).head(10)

top_recommendation = Weighed_avg.sort_values(by='weighted_avg',ascending=False)
#validate this manually by seeing past history of current user and recommended movie list..

# as we see below current user mostly love to watch comedy/drama and other 



#first merge MovieId in movie name and genre so we can classify based on genre and name..

Rating_Movie = Ratings.merge(Movies, left_on='movieId',right_on='movieId')

Watched_Genre = pd.DataFrame(Rating_Movie[Rating_Movie.userId==359].groupby('genres').size(),columns=['Count'])

Watched_Genre.sort_values(by='Count',ascending=False).head(10)
#our recommendation is matching with user past history ...

top_recommendation['movieId'] = top_recommendation.index

Movies_recommended = Movies.merge(top_recommendation, left_on='movieId',right_on='movieId')

Watched_recommended_Genre = pd.DataFrame(Movies_recommended.groupby('genres').size(),columns=['Count'])

Watched_recommended_Genre.sort_values(by='Count',ascending=False).head(10)

#item wise similarity with cosine...

item_similarity = cosine_similarity(RatingMat.T)

item_sim_df = pd.DataFrame(item_similarity,index=RatingMat.columns,columns=RatingMat.columns)

item_sim_df.head(5)



#item wise similarity with pearson coorelation...

# item_similarity = 1-pairwise_distances(RatingMat.T, metric="correlation")

# item_sim_df = pd.DataFrame(item_similarity,index=RatingMat.columns,columns=RatingMat.columns)

# item_sim_df.head(5)

'''this is just example how we pick similarity for any Id later we will take similarity for all movie 1 by 1 and 

then check with specific user.. 

'''

curr_movie_similarity = pd.DataFrame(item_sim_df.loc[5])

curr_movie_similarity.rename(columns={5:'similarity'},inplace=True)

curr_movie_similarity.head()
'''

In this we have to first calculate take similarity of movie ex similarity of movie 5 w.r.t to all and 

then multiply with user1 all movie rating rating then divide by all sum of similarity so we get weighted avg for 

movie 5. similarly for all we have to calculate. 

'''



#formulas :- sum(w[1]*movie1+w[2]*movie2.....w[n]*movieN) / w[1]+w[2]....w[n] 

Weighed_movie_avg = []

 

for movieId in item_sim_df.index:

    

    #extract similarity of particular movieId with respect to others.

    curr_movie_similarity = pd.DataFrame(item_sim_df.loc[movieId])

    

    #calculate sum of all similarity

    Total_movie_weight = np.sum(item_sim_df.loc[movieId])

    

    #calculate dot product with user 1 to similarity of particular movieId and divide by total weight.

    Weighed_movie_avg.append(np.dot(RatingMat.loc[359],curr_movie_similarity) \

                             /(np.abs(Total_movie_weight)))

    

#above loop works for all movie 1 by 1 and gives weighted avg value for all movieId
#convert weighted avg of all movie into frame and set column and index name..

Weighed_movie_avg = pd.DataFrame(Weighed_movie_avg)

Weighed_movie_avg.index = RatingMat.columns

Weighed_movie_avg.columns = ['weighted_avg']

Weighed_movie_avg.head()
#sort top 10 best weighted avg score movie fot userk..

Weighed_movie_avg.sort_values(by='weighted_avg',ascending=False).head(10)

top_10_recommendation_itemwise = Weighed_movie_avg.sort_values(by='weighted_avg',ascending=False).head(10)
#validate this manually by seeing past history of current user and recommended movie list..

# as we see below current user mostly love to watch Mystrey/Thriller abd some 



#first merge MovieId in movie name and genre so we can classify based on genre and name..

Rating_Movie = Ratings.merge(Movies, left_on='movieId',right_on='movieId')

Rating_Movie[Rating_Movie.userId==359].head(10)
#our recommendation is matching with user past history ... 

Rating_Movie.loc[top_10_recommendation_itemwise.index]