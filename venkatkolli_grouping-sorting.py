import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.groupby("taster_twitter_handle").count()
# check_q1(common_wine_reviewers)
# Your code here
best_wine = reviews.groupby("price").price.max()
best_wine
# Your code here
reviews.groupby(["variety"]).price.agg([min, max])
# check_q3(wine_price_extremes)
# Your code here
reviews.groupby(["taster_name"]).points.mean()
# check_q4(reviewer_mean_rating)
reviews.groupby(["variety"]).price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
# wine_price_range =
# check_q5(wine_price_range)
reviews['n']=0
reviews.groupby(["country", "variety"]).n.count().sort_values(ascending=False)
# country_variety_pairs = _____
# check_q6(country_variety_pairs)