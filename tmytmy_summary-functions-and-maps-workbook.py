import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
median= reviews.points.median()# Your code here
check_q1(median)
ans = reviews.country.unique()
check_q2(ans)
ans = reviews.country.value_counts()
check_q3(ans)
median = reviews.price.median()
check_q4(reviews.price.map(lambda x : x-median))
median = reviews.price.median()
reviews.price.apply(lambda x : x-median)
answer_q5()
reviews.title[(reviews.points / reviews.price).idxmax()]
#check_q6(best)
#answer_q6()
#reviews.loc[(reviews.points / reviews.price).argmax()].title
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()