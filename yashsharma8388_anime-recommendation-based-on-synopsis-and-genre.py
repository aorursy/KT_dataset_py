import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
# Reading Datasets
anime_title = pd.read_csv('../input/datatitle-all-share-new.csv', sep = "|")
anime_genre = pd.read_csv('../input/datagenre-all-share-new.csv', sep = "|")
anime_syn = pd.read_csv('../input/datasynopsis-all-share-new.csv', sep = "|")
print (f"Shape of Anime Title: {anime_title.shape}")
print (f"Shape of Anime Synopsis: {anime_syn.shape}")
print (f"Shape of Anime Genre: {anime_genre.shape}")
anime_title.head()
anime_genre.head()
anime_syn.head()
anime = pd.merge(anime_title, anime_syn, on = ['Anime_ID'])
anime = pd.merge(anime, anime_genre, on = ['Anime_ID'])
anime.head()
anime.info()
# Genres are in the format action;ecchi;adventure etc
# Clean (or Parse) the Genres for each anime
def clean_genres(x):
    if isinstance(x['Genres'], str):
        gen = x['Genres'].replace(";", ", ")
    else:
        gen = ""
        
    return gen
# Replace NaN with "" i.e with empty string in the synopsis
anime['Synopsis'].fillna("", inplace = True)

anime['Genres'] = anime.apply(clean_genres, axis = 1)
# Construct Similarity Matrix for Synopsis and Genres

# 1. Get the Indicies
indices = pd.Series(anime.index, index = anime['Anime_name'])
# 2 Setup the TfidfVect and  CountVec
tfidf = TfidfVectorizer(stop_words = "english")
countvec = CountVectorizer(stop_words = "english")
# 3. Get Matrix for both
tfidf_mat = tfidf.fit_transform(anime['Synopsis'])
countvect_mat = countvec.fit_transform(anime['Genres'])
# 4. Cosine Similarity Score
syn_similarity = linear_kernel(tfidf_mat, tfidf_mat)
genre_similarity = linear_kernel(countvect_mat, countvect_mat)
# 5. Get Recommendation
def getRecommendation(title):
    
    # Get the Index of the Anime.
    idx = indices[title]
    
    # We have 2 Similarity Metrics
    ## 1. Synopsis Similarity
    ## 2. Genre Similarity
    
    score_1 = list(enumerate(syn_similarity[idx]))
    score_2 = list(enumerate(genre_similarity[idx]))
    
    # Sort the scores in reverse order
    score_1 = sorted(score_1, key = lambda x: x[0], reverse = False)
    score_2 = sorted(score_2, key = lambda x: x[0], reverse = False)    
    
    # Average of the two Similarity (Cosine) Scores
    combined_score = [(idx, (sc_1 + sc_2) / 2) for (idx, sc_1), (_, sc_2) in zip(score_1, score_2)]
    
    # Sorting the Combined Score.
    combined_score = sorted(combined_score, key = lambda x: x[1], reverse = True)
    
    # Get ID of Top 10 Similar Animes
    anime_ids = [i[0] for i in combined_score[1:11]]
    
    # Returning the Top Anime Names.
    return anime['Anime_name'].iloc[anime_ids]
def showRecommendation(anime_name):
    g = getRecommendation(title = anime_name)

    print (f"Your Anime: {anime_name}\n\nRecommended Anime for you: \n")
    for i, name in g.iteritems():
        print (name)
showRecommendation(random.choice(anime['Anime_name']))
showRecommendation(random.choice(anime['Anime_name']))
