import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
data = reviews.points.median()
check_q1(data)
data= reviews.country
check_q2(data)
data = reviews.country.value_counts()
check_q3(data)
t = reviews.price.median()
data = reviews.price.map(lambda x: x-t)
check_q4(data)
data = reviews.loc[(reviews.points / reviews.price).argmax()].title
check_q5(data)
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()