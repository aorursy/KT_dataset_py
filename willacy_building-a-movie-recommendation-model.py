import numpy as np

import pandas as pd

from fastai.tabular import *

from fastai.collab import *
movies = pd.read_csv('/kaggle/input/movietweetings/movies.dat', delimiter='::', engine='python', header=None, names = ['Movie ID', 'Movie Title', 'Genre'])

users = pd.read_csv('/kaggle/input/movietweetings/users.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Twitter ID'])

ratings = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Movie ID', 'Rating', 'Rating Timestamp'])

movies.head()
users.head()
ratings.head()
for i in [movies, users, ratings]:

    print(i.shape)
df = ratings.merge(movies[['Movie ID','Movie Title']], on='Movie ID')
df = df.rename(columns={'User ID':'userID','Movie ID':'movieID','Rating':'rating','Rating Timestamp':'timestamp', 'Movie Title': 'title'})

df.head()
df.rating = df.rating/2.0
data = CollabDataBunch.from_df(df, seed=42, valid_pct=0.1, item_name='title')
data.show_batch()
y_range = [0,5.5]
learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3)
learn.save('dotprod')
learn.load('dotprod');
g = df.groupby('title')['rating'].count()

top_movies = g.sort_values(ascending=False).index.values[:1000]

top_movies[:10]
movie_bias = learn.bias(top_movies, is_item=True)

movie_bias.shape
mean_ratings = df.groupby('title')['rating'].mean()*2

movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_movies,movie_bias)]
item0 = lambda o:o[0]
sorted(movie_ratings, key=item0)[:15]
df2 = df.copy()

df2.rating = df2.rating*2
df2[df2.title == 'The Thin Red Line (1998)'].groupby('rating')['title'].count()
sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]
df2[df2.title == 'Be Somebody (2016)'].groupby('rating')['title'].count()
movie_w = learn.weight(top_movies, is_item=True)

movie_w.shape
movie_pca = movie_w.pca(3)

movie_pca.shape
fac0,fac1,fac2 = movie_pca.t()

movie_comp = [(f, i) for f,i in zip(fac0, top_movies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
movie_comp = [(f, i) for f,i in zip(fac1, top_movies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
movie_comp = [(f, i) for f,i in zip(fac2, top_movies)]

sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
idxs = np.random.choice(len(top_movies), 75, replace=False)

idxs = list(range(75))

X = fac0[idxs]

Y = fac1[idxs]

plt.figure(figsize=(15,15))

plt.scatter(X, Y)

for i, x, y in zip(top_movies[idxs], X, Y):

    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)

plt.xlabel("<------- Movies like Movie43 and Scary Movie 5                     [fac0]                            Gritty Heros------>")

plt.ylabel("<------- Big Action                                [fac1]                                         Thriller/Horrors------>")

plt.title("75 Top Movies")

plt.show()
df.userID.value_counts()[:10]
learn.export()
learn = load_learner('/kaggle/input/export/')
h = df.groupby('movieID')['rating'].count()

all_films = h.sort_values(ascending=False).index.values[:10000]
def get_top_suggested(user_id):

    user_films = df[df.userID == user_id].movieID.sort_values().values

    unseen_films = [i for i in all_films if i not in user_films]

    user_df = pd.Series(unseen_films, name='movieID').to_frame()

    user_df['userID'] = user_id

    user_df = user_df.reindex(columns=['userID','movieID'])

    user_df = user_df.merge(df[['movieID','title']], on='movieID')

    user_df = user_df.groupby('title').mean().reset_index()

    learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(user_df, cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))

    preds = learn.get_preds(ds_type=DatasetType.Test)

    user_df['rating'] = preds[0]*2

    user_df['rating'] = round(user_df['rating'],1)

    return user_df.sort_values(by='rating', ascending=False).reset_index()[['title','rating']].head(20)
get_top_suggested(24249)
def rating_comparison(user_id):

    learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(df[df.userID == 24249], cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))

    preds = learn.get_preds(ds_type=DatasetType.Test)

    df2 = df[df.userID == 24249].copy()

    df2['prediction'] = preds[0]*2

    df2['rating'] = df2['rating']*2

    df2['diff'] = abs(df2['prediction'] - df2['rating'])

    return df2.sort_values(by='prediction', ascending=False).reset_index()[['title','rating','prediction','diff']]
rating_comparison(24249).head(20)
learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(df, cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))

preds = learn.get_preds(ds_type=DatasetType.Test)
df['prediction'] = preds[0]*2

df['difference'] = abs(df['prediction'] - df['rating']*2)

df2 = df.groupby('userID')[['difference']].agg(['count','mean'])

df2.difference.plot(x='mean',y='count',kind='scatter',figsize=(20,10))