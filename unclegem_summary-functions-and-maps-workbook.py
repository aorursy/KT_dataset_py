import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()

price_median = reviews.price.median()
reviews.price.map(lambda price: price - price_median)
reviews['price']
reviews.loc[(reviews.points / reviews.price).idxmax()].title
t = reviews.description.map(lambda t: "tropical" in t).value_counts()
f = reviews.description.map(lambda f: "tropical" in f).value_counts()
pd.Series([t[True], f[True]], ["tropical", "fruity"])
reviews.country[reviews.country.notnull()] + "-" + reviews.variety[reviews.variety.notnull()]