import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')
anime = pd.read_csv("../input/anime.csv")
anime.head()
# only select tv show and movie
print(anime.shape)
anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
print(anime.shape)
# only select famous anime, 75% percentile
m = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= m)]
anime.shape
rating = pd.read_csv("../input/rating.csv")
rating.head()
rating.shape
rating.loc[rating.rating == -1, 'rating'] = np.NaN
rating.head()
anime_index = pd.Series(anime.index, index=anime.name)
anime_index.head()
joined = anime.merge(rating, how='inner', on='anime_id')
joined.head()
joined = joined[['user_id', 'name', 'rating_y']]

pivot = pd.pivot_table(joined, index='name', columns='user_id', values='rating_y')
pivot.head()
pivot.shape
pivot.dropna(axis=1, how='all', inplace=True)
pivot.head()
pivot.shape
pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.head()
# fill NaN with 0
pivot_norm.fillna(0, inplace=True)
pivot_norm.head()
# convert into dataframe to make it easier
item_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index, columns=pivot_norm.index)
item_sim_df.head()
def get_similar_anime(anime_name):
    if anime_name not in pivot_norm.index:
        return None, None
    else:
        sim_animes = item_sim_df.sort_values(by=anime_name, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=anime_name, ascending=False).loc[:, anime_name].tolist()[1:]
        return sim_animes, sim_score
animes, score = get_similar_anime("Steins;Gate")
for x,y in zip(animes[:10], score[:10]):
    print("{} with similarity of {}".format(x, y))
# predict the rating of anime x by user y
def predict_rating(user_id, anime_name, max_neighbor=10):
    animes, scores = get_similar_anime(anime_name)
    anime_arr = np.array([x for x in animes])
    sim_arr = np.array([x for x in scores])
    
    # select only the anime that has already rated by user x
    filtering = pivot_norm[user_id].loc[anime_arr] != 0
    
    # calculate the predicted score
    s = np.dot(sim_arr[filtering][:max_neighbor], pivot[user_id].loc[anime_arr[filtering][:max_neighbor]]) \
            / np.sum(sim_arr[filtering][:max_neighbor])
    
    return s
predict_rating(3, "Steins;Gate")
predict_rating(3, "Cowboy Bebop")
# recommend top n_anime for user x based on item collaborative filtering algorithm
def get_recommendation(user_id, n_anime=10):
    predicted_rating = np.array([])
    
    for _anime in pivot_norm.index:
        predicted_rating = np.append(predicted_rating, predict_rating(user_id, _anime))
    
    # don't recommend something that user has already rated
    temp = pd.DataFrame({'predicted':predicted_rating, 'name':pivot_norm.index})
    filtering = (pivot_norm[user_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

    # recommend n_anime anime
    return anime.loc[anime_index.loc[temp.name[:n_anime]]]
get_recommendation(3)
get_recommendation(5)
