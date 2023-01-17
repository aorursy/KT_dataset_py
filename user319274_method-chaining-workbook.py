import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
my_ans7_q1 = chess_games.winner.value_counts()/len(chess_games)
check_q1(my_ans7_q1)
my_ans7_q2 = chess_games.opening_name.map(lambda n: n.split(':')[0].split('|')[0].split('#')[0].strip()).value_counts()
check_q2(my_ans7_q2)
chess_games['n'] = 0
my_ans7_q3 = chess_games.groupby(['white_id', 'victory_status']).n.count().reset_index()
check_q3(my_ans7_q3)
top20_index = chess_games.white_id.value_counts().head(20).index

my_ans7_q4 = (chess_games
    .assign(n = 0)
    .groupby(['white_id', 'victory_status'])
    .n
    .count()
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(top20_index)]))
check_q4(my_ans7_q4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
my_ans7_q5 = kepler.assign(n = 0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
check_q5(my_ans7_q5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
my_ans7_q6 = wine_reviews.points.dropna().map(lambda p: (p - 80)/4).value_counts().rename_axis("Wine Ratings").sort_index(ascending=True)
check_q6(my_ans7_q6)
my_ans7_q7 = (ramen_reviews['Stars']
      .replace('Unrated', None)
      .dropna()
      .astype('float64')
      .value_counts()
      .rename_axis("Ramen Ratings")
      .sort_index(ascending=True))

check_q7(my_ans7_q7)
my_ans7_q8 = (ramen_reviews['Stars']
      .replace('Unrated', None)
      .dropna()
      .astype('float64')
      .map(lambda p: int(p) if p/(int(p)+0.5) < 1 else int(p)+0.5)
      .value_counts()
      .rename_axis("Ramen Ratings")
      .sort_index(ascending=True))

check_q8(my_ans7_q8)