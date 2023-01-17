import numpy as np

import pandas as pd

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel



import surprise

from surprise import Dataset

from surprise import Reader

from surprise import SVDpp

from surprise import model_selection

from surprise.model_selection import cross_validate, GridSearchCV
df_rating = pd.read_csv('../input/rating.csv')

df_anime = pd.read_csv('../input/anime.csv')

print("Rating's shape: {}".format(df_rating.shape))

print("Anime's shape: {}".format(df_anime.shape))
print(df_rating.head())

print(df_rating.rating.unique())
df_anime.head()
# Remove null values (-1)

df_rating = df_rating[df_rating.rating != -1]

df_rating.shape
df_anime_rec1 = df_anime.drop(['genre', 'type', 'episodes', 'members'], axis=1)

df_rating_rec1 = df_rating.drop(['rating', 'user_id'], axis=1)

df_anime_rating_total = df_anime_rec1.merge(df_rating_rec1, how='inner', on='anime_id')

df_anime_rating_total.head()
df_anime_rating_total = df_anime_rating_total.dropna()

df_anime_rating_number = df_anime_rating_total.groupby(['anime_id'], as_index=False)['rating'].count()

df_anime_rating_number = df_anime_rating_number.rename(columns={'rating': 'rating number'})

df_anime_rating_number.head()
df_anime_rating_total = df_anime_rating_total.merge(df_anime_rating_number, on='anime_id', how='inner')

df_anime_rating_total.head()
df_anime_rating = df_anime_rating_total.drop_duplicates(keep='first')

df_anime_rating = df_anime_rating.reset_index()

df_anime_rating = df_anime_rating.drop(['index'], axis=1)

df_anime_rating.head()
warnings.filterwarnings('ignore')



C = df_anime_rating['rating'].mean()

m = df_anime_rating['rating number'].quantile(0.9)

df_anime_rating_recommend = df_anime_rating[df_anime_rating['rating number'] >= m]

df_anime_rating_recommend['scoring'] = df_anime_rating_recommend['rating number']*df_anime_rating_recommend['rating']/(df_anime_rating_recommend['rating number']+m) + m*C/(df_anime_rating_recommend['rating number']+m)

df_anime_rating_recommend.head()
df_anime_rating_recommend = df_anime_rating_recommend.sort_values('scoring', ascending=False)

df_anime_rating_recommend.head(10)
# Show detailed information about those most popular movies

df_anime_popular = pd.DataFrame({'anime_id':[], 'name':[], 'genre':[], 'type':[], 'episodes':[], 'rating':[], 'members':[]})



df_anime_popular_name = df_anime_rating_recommend.head(10)['name']

for name in df_anime_popular_name:

    df_anime_popular = df_anime_popular.append(df_anime[df_anime['name'] == name])

        

df_anime_popular = df_anime_popular.reset_index()

df_anime_popular = df_anime_popular.drop(['index'], axis=1)

df_anime_popular
df_anime.isna().sum()
df_anime_cbrs = df_anime.dropna(axis=0)

df_anime_cbrs = df_anime_cbrs.reset_index()

df_anime_cbrs.isna().sum()
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df_anime_cbrs['genre'])



# Calculate cosine similarities

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_anime_cbrs.index, index=df_anime_cbrs['name']).drop_duplicates()



def get_recommendations_cb(title, cosine_sim=cosine_sim):

    index = indices[title]

    sim_scores = list(enumerate(cosine_sim[index]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df_anime['name'].iloc[movie_indices]
get_recommendations_cb('Fullmetal Alchemist: Brotherhood')
df_rating.head()
data = df_rating['rating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / df_rating.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )



layout = dict(title = 'Distribution Of {} moive-ratings'.format(df_rating.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df = df_rating.iloc[:50000,].reset_index()

df = df.drop(['index'], axis=1)

df.head()
df['rating'].unique()
recmodel = SVDpp()

reader = Reader(rating_scale=(1,10))

df_rating_rec = Dataset.load_from_df(df, reader)

recmodel.fit(df_rating_rec.build_full_trainset()) 

cross_validate(recmodel, df_rating_rec, measures=['RMSE', 'MAE'], cv=5, verbose=True)
anime_id = df['anime_id'].unique()

# just take user1 for example

anime_id1 = df.loc[df['user_id'] == 1, 'anime_id']

anime_id_to_pred = np.setdiff1d(anime_id, anime_id1)
testset = [[1, anime_id, 10] for anime_id in anime_id_to_pred]

user_id1_pred = recmodel.test(testset)

df_pred = pd.DataFrame(user_id1_pred)

df_pred.head()
df_pred = df_pred.rename(columns={'uid': 'user_id', 'iid': 'anime_id', 'est': 'predicted rating'})

df_pred = df_pred.drop(['r_ui', 'details'], axis=1)

df_pred.head()
df_pred = df_pred.sort_values('predicted rating', ascending=False)

df_pred_anime_id = df_pred.head(10)['anime_id']



df_recommendation = pd.DataFrame({'anime_id':[], 'name':[], 'genre':[], 'type':[], 'episodes':[], 'rating':[], 'members':[]})



for anime_id in df_pred_anime_id:

    df_recommendation = df_recommendation.append(df_anime[df_anime['anime_id'] == anime_id])

        

df_recommendation = df_recommendation.reset_index()

df_recommendation = df_recommendation.drop(['index'], axis=1)

df_recommendation['anime_id'] = df_recommendation['anime_id'].astype('int')

df_recommendation
def get_recommendations_cf(user_id, num_recommendations):

    """Provide recommendations for specific user with the number they want to show.

    """

    anime_id = df['anime_id'].unique()

    anime_id_user = df.loc[df['user_id'] == user_id, 'anime_id']

    anime_id_to_pred = np.setdiff1d(anime_id, anime_id_user)

    testset = [[user_id, anime_id, 10] for anime_id in anime_id_to_pred]

    user_id_pred = recmodel.test(testset)

    df_pred = pd.DataFrame(user_id_pred)

    

    df_pred = df_pred.sort_values('est', ascending=False)

    df_pred_anime_id = df_pred.head(num_recommendations)['iid']



    df_recommendation = pd.DataFrame({'anime_id':[], 'name':[], 'genre':[], 'type':[], 'episodes':[], 'rating':[], 'members':[]})



    for anime_id in df_pred_anime_id:

        df_recommendation = df_recommendation.append(df_anime[df_anime['anime_id'] == anime_id])

        

    df_recommendation = df_recommendation.reset_index()

    df_recommendation = df_recommendation.drop(['index'], axis=1)

    df_recommendation['anime_id'] = df_recommendation['anime_id'].astype('int')

    return df_recommendation
get_recommendations_cf(1, 15)
df_anime_popular
get_recommendations_cb('Fullmetal Alchemist: Brotherhood')
get_recommendations_cf(2, 5)