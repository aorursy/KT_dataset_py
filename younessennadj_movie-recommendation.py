# Define the libraries and imports

# Panda

import pandas as pd

#mat plot

import matplotlib.pyplot as plt

#Sea born

import seaborn as sns

#Num py

import numpy as np

#Word Count

import wordcloud as wc



#Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity



#CountVectorizer IMPORT

from sklearn.feature_extraction.text import CountVectorizer



import os

import warnings

warnings.filterwarnings('ignore')
# Load data from the path to the dataSet and force imdbRate to be numeric

def load_dataset_with_rate(dataSet_path):

    data = pd.read_csv(dataSet_path)

    data['imdbRate'] = pd.to_numeric(data['imdbRate'],errors='coerce')

    return data



# Load data from the path to the dataSet

def load_dataset(dataSet_path):

    data = pd.read_csv(dataSet_path)

    return data



# transform columns that have pipe as a separator

def tranform_columns(df):



    df["actors"]= df["actors"].str.split("|", n = 10, expand = False) 



    df["writers"]= df["writers"].str.split("|", n = 10, expand = False) 



    df["genres"]= df["genres"].str.split("|", n = 10, expand = False)     
# Load dataset

df= load_dataset_with_rate("../input/movies.csv")



df.head(10)
tranform_columns(df)



df.head(10)
# Load dataset

dr= load_dataset("../input/ratings-1m.csv")



dr.head(10)
df['imdbRate'].plot(kind='box', subplots=True)
df['imdbRate'].hist()
p = dr.groupby('rating')['rating'].agg(['count'])



# get movie count

movie_count = df['movieId'].count()



# get customer count

cust_count = dr['userId'].nunique() 



# get rating count

rating_count = dr['userId'].count()



ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))

plt.title('Total pool: {:,} Movies, {:,} users, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)

plt.axis('off')



for i in range(1,11):

    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')
userRatingsAggr = dr.groupby(['userId']).agg({'rating': [np.size, np.mean]})

userRatingsAggr.reset_index(inplace=True)  # To reset multilevel (pivot-like) index

userRatingsAggr['rating'].plot(kind='box', subplots=True)
movieRatingsAggr = dr.groupby(['movieId']).agg({'rating': [np.size, np.mean]})

movieRatingsAggr.reset_index(inplace=True)

movieRatingsAggr['rating'].plot(kind='box', subplots=True)
#Plot the count of the movies by years

def count_pie(series):

    counts=series.value_counts()

    counts=counts/counts.sum()

    labels=['' if num<0.01 else str(year) for (year,num) in counts.items()]

    f, ax = plt.subplots(figsize=(8, 8))

    explode = [0.02 if counts.iloc[i] < 100 else 0.001 for i in range(counts.size)]

    plt.pie(counts,labels=labels,autopct=lambda x:'{:1.0f}%'.format(x) if x > 1 else '',explode=explode)

    plt.show()



count_pie(df.date.dropna().apply(lambda x:str(int(x)//10*10)+'s'))    
def plotjoint(data,x,y,xlim=None,ylim=None,xscale=None,yscale=None):

    sns.set(style=None,font_scale=2)

    grid=sns.jointplot(data[x],data[y],kind="hex",color="#4CB391",height=15,ratio=10,xlim=xlim,ylim=ylim)



plotjoint(df,"date","imdbRate",xlim=(1900,2020),ylim=(4,8.5))
# numbers of movies of different genres

def count_multiple(df,column):    

    multiple= df[column]

    multiple= multiple.apply(pd.Series.value_counts)

    multiple = multiple.sum(axis = 0, skipna = True)

    multiple = multiple.sort_values(ascending=False).nlargest(20)

    return multiple
genres_count = count_multiple(df,'genres')

genres_count
f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation=85, fontsize=15)

genres_count.plot.bar()

plt.show()
#Wordcloud of genres and keywords

def multi_wordcloud(series):

    w=wc.WordCloud(background_color="white",margin=20,width=800,height=600,prefer_horizontal=0.7,max_words=20,scale=2)

    w.generate_from_frequencies(series)

    f, ax = plt.subplots(figsize=(16, 8))

    plt.axis('off')

    plt.imshow(w)

    plt.show()

#Apply the word cloud on the genres

multi_wordcloud(genres_count)
mean_rate_per_type = df.groupby('type')['imdbRate'].count().sort_values(ascending=False).head(15)

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation=85, fontsize=15)

p = sns.barplot(x = mean_rate_per_type.index, y = mean_rate_per_type.values)

p = plt.xticks(rotation=90)

plt.show()
count_rate_per_director = df.groupby('director')['imdbRate'].count().sort_values(ascending=False).head(15)

multi_wordcloud(count_rate_per_director)

count_rate_per_director = df.groupby('director')['imdbRate'].mean().sort_values(ascending=False).head(15)

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation=85, fontsize=15)

p = sns.barplot(x = mean_rate_per_type.index, y = mean_rate_per_type.values)

p = plt.xticks(rotation=90)

plt.show()
# Define categorical duration 

def categorize_duration(df,column):

    bins = (5,60,120,300,1000)

    group_names = ['short','Medium','Long','VeryLong']

    categories = pd.cut(df[column],bins,labels=group_names)

    df2 = df.copy()

    df2[column]=categories

    return df2



#Apply the function

df_duration_categories = categorize_duration(df,'duration')



#plot the graph

mean_rate_per_duration = df_duration_categories.groupby('duration')['imdbRate'].mean().sort_values(ascending=False)

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation=85, fontsize=15)

p = sns.barplot(x = mean_rate_per_duration.index, y = mean_rate_per_duration.values)

p = plt.xticks(rotation=90)

plt.show()
C= df['imdbRate'].mean()

C
m= df['imdbRateCount'].quantile(0.9)

m
q_movies = df.copy().loc[df['imdbRateCount'] >= m]

q_movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['imdbRate']

    R = x['imdbRateCount']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)



#Print the top 15 movies

q_movies[['title', 'imdbRate', 'imdbRateCount', 'score']].head(10)
#The next step would be to convert the 

#instances into lowercase and strip all the spaces between them. 

#This is done so that our vectorizer doesn't count the Johnny of"Johnny Depp"and"Johnny Galecki" as the same.



# Function to convert all strings to lower case and strip names of spaces



# we have list like Genres and actors  and we have srings like director

def clean_data(x):

    if isinstance(x, list):  

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''



# clean remove spaces

features = ['director', 'genres', 'actors','writers']



for feature in features:

    df[feature] = df[feature].apply(clean_data)

df.head(10)    
def create_soup(x):

    return ' '.join(x['genres']) + ' ' + ' '.join(x['actors']) + ' ' + x['director'] + ' ' + ' '.join(x['writers'])

df['soup'] = df.apply(create_soup, axis=1)

df['soup'].head()
# Import CountVectorizer and create the count matrix

# so we want to make cosine similarity but there are two questions how we can make it and why we use it specially 

# at the first we have here 236056 movies so for each movie we want to know how many times the word from

# soup column appeared in each movie how we can make that ?

# by using CountVectorizer steps:

# 1-create instance of CountVectorizer

# 2-Fit is to (learn vocabularies) learning here means we want to know the word and repeated count for it in whole files

# like : mandiargues': 3705



# 3 - transform - > encode all movies documents as vectors so you will have each movie with all words so you can know 

# how many times this word appeared in each movie 









#Stopwords are words which do not contain enough significance to

#be used without our algorithm. We would not want these words taking up space i



count = CountVectorizer(stop_words='english')

vocab = count.fit(df['soup'])

dict(list(vocab.vocabulary_.items())[0:10])
count_matrix = vocab.transform(df['soup'])

print(count_matrix)
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations_names(df,id_movie):

    cosine_sim = cosine_similarity(count_matrix[df.loc[df['movieId']==id_movie].index[0],] , count_matrix)





    

    # Get the index of the movie that matches the title

    idx = 0



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return df['title'].iloc[movie_indices].tolist()
get_recommendations_names(df,1)
#Surprise import

from surprise import Dataset,Reader,SVD,SVDpp,BaselineOnly,NMF,NormalPredictor,CoClustering,KNNBaseline,KNNWithMeans,KNNBasic,SlopeOne

from surprise.model_selection import cross_validate,train_test_split

from surprise.model_selection import GridSearchCV as GridSearchCV_surprise 

from surprise import accuracy



#Timer

from datetime import datetime
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\nTime taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

        return (datetime.now() - start_time).total_seconds()



def train_and_time(algo,trainset,testset):

    start_time = timer(None)



    # Train the algorithm on the trainset, and predict ratings for the testset

    algo.fit(trainset)



    score_time.append(timer(start_time))



    predictions = algo.test(testset)



    # Then compute RMSE

    score_rmse.append(accuracy.rmse(predictions))

    # Then compute MAE

    score_mae.append(accuracy.mae(predictions))
score_rmse = []

score_mae=[]

score_time=[]

score_algos = []

predictions =''
# path to dataset file

file_path = os.path.expanduser('../input/ratings-1m.csv')



# As we're loading a custom dataset, we need to define a reader. In the

# movielens-1M dataset, each line has the following format:

# 'user item rating timestamp', separated by ',' characters.

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)



data = Dataset.load_from_file(file_path, reader=reader)



# sample random trainset and testset

# test set is made of 20% of the ratings.

trainset, testset = train_test_split(data, test_size=.20)
# We'll use the famous SVD algorithm.

score_algos.append('SVD')

svd = SVD()

train_and_time(svd,trainset,testset)
def get_related_movies(data,uid):

    #Get a list of all movies ids

    iids= data['movieId'].unique()

    #Get a list of iids that uid has rated

    iids_of_uid = data.loc[data['userId']==uid,'movieId']

    #Remove the iids that uid has rated from the list of all movie ids

    iids_to_pred = np.setdiff1d(iids,iids_of_uid)

    

    return iids_to_pred



def get_top_movies_prediction_for_user(algo,uid,iids_to_pred):

    predictions=[]

    for iid in iids_to_pred:

        predictions.append(algo.predict(str(uid),str(iid),4.0))

    return predictions



def get_n_top_movies(predictions,iids_to_pred,rec_num):

    pred_ratings = np.array([pred.est for pred in predictions])

    rec_items = []

    for i in range(rec_num):

        i_max=pred_ratings.argmax()

        rec_items.append(iids_to_pred[i_max])

        pred_ratings=np.delete(pred_ratings,i_max)

    return rec_items
uid = 57

iids_to_pred = get_related_movies(dr,uid)

predictions = get_top_movies_prediction_for_user(svd,uid,iids_to_pred)

print(get_n_top_movies(predictions,iids_to_pred,10))