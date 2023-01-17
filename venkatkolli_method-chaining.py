import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
chess_games["winner"].value_counts()/len(chess_games) 
chess_games["winner"].value_counts()
chess_games["winner"]
chess_games["opening_name"].map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.count().sort_values(ascending=False).reset_index()
def top_20(df):
    df.assign(n=0).groupby(['white_id', 'victory_status']).n.count().sort_values(ascending=False).reset_index()
    return df.head(20)
(chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.count().sort_values(ascending=False).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().index)])).head(20)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count().reset_index()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
((wine_reviews['points'].dropna()-80)/4).value_counts().sort_index().rename_axis("Wine Ratings")
ramen_reviews['Stars'].replace('Unrated', None).dropna().astype('float64').value_counts().rename_axis('Ramen Reviews').sort_index()
ramen_reviews['Stars'].replace('Unrated', None).astype('float64').map(lambda x: int(x) if x - int(x)<0.5 else int(x)+0.5).value_counts().sort_index()