%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from surprise import Reader, Dataset, SVD, evaluate



import warnings; warnings.simplefilter('ignore')
md = pd. read_csv('../input/movies_metadata.csv')

md.head()
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
C = md['vote_average'].mean()

m= md['vote_count'].quantile(0.95)

print("C is %f, and m is %d"%(C,m))
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())]

qualified.shape
def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)

qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

qualified['score'] = qualified.apply(weighted_rating, axis=1)

qualified = qualified.sort_values('score', ascending=False)

qualified.head(10)
links = pd.read_csv('../input/links_small.csv')

links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

print (md[pd.to_numeric(md['id'], errors='coerce').isnull()])
md = md.drop([19730, 29503, 35587])

md['id'] = md['id'].astype('int')

def get_recommendations(title, cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
credits = pd.read_csv('../input/credits.csv')

keywords = pd.read_csv('../input/keywords.csv')

keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

md = md.merge(credits, on='id')

md = md.merge(keywords, on='id')

smd1 = md[md['id'].isin(links)]



features = ['cast', 'crew', 'keywords']

for feature in features:

    smd1[feature] = smd1[feature].apply(literal_eval)
def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan

smd1['director'] = smd1['crew'].apply(get_director)

smd1.head()
def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names

    return []

features = ['cast', 'keywords']

for feature in features:

    smd1[feature] = smd1[feature].apply(get_list)

smd1[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''

# Apply clean_data function to your features.

features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    smd1[feature] = smd1[feature].apply(clean_data)

smd1['director'] = smd1['director'].apply(lambda x: [x,x, x])

smd1.head(3)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])  + ' '.join(x['director']) + ' '.join(x['genres'])

smd1['soup'] = smd1.apply(create_soup, axis=1)

smd1[['title', 'cast', 'director', 'keywords', 'genres', 'soup']].head(3)
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = count.fit_transform(smd1['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)



smd1 = smd1.reset_index()

titles = smd1['title']

indices = pd.Series(smd1.index, index=smd1['title'])



indices.head()

get_recommendations('Toy Story',cosine_sim)
reader = Reader()

ratings = pd.read_csv('../input/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

data.split(n_folds=5)

svd = SVD()

evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

svd.train(trainset)

ratings[ratings['userId'] == 1]
svd.predict(1, 302)
def convert_int(x):

    try:

        return int(x)

    except:

        return np.nan

    

id_map = pd.read_csv('../input/links_small.csv')[['movieId', 'tmdbId']]

id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)

id_map.columns = ['movieId', 'id']

id_map = id_map.merge(smd1[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

def hybrid(userId, title):

    idx = indices[title]

    tmdbId = id_map.loc[title]['id']

    #print(idx)

    movie_id = id_map.loc[title]['movieId']

    

    sim_scores = list(enumerate(cosine_sim[int(idx)]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]

    movies = smd1.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]

    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)

    movies = movies.sort_values('est', ascending=False)

    return movies.head(10)

hybrid(1, 'Avatar')

hybrid(500, 'Avatar')