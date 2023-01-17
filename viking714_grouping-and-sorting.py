import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head(10)
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
check_q1(reviews.groupby("taster_twitter_handle").taster_twitter_handle.count())
# Your code here
# best_wine = ______
# check_q2(best_wine)
check_q2(reviews.groupby("price").points.max())
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
check_q3(reviews.groupby('variety').price.agg(['min','max']))
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
check_q4(reviews.groupby('taster_name').points.mean())
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
check_q5(reviews.groupby('variety').price.agg(['min','max']).sort_values(by=['min','max'],ascending=False))
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews['n'] = 1
check_q6(reviews.groupby(['country','variety']).n.count().sort_values(ascending = False))