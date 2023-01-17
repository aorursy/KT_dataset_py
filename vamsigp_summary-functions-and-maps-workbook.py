import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
# val = np.median(reviews['points'])
# print(reviews.points.median)
# print('##')
# print(val)
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
med = reviews.price.median()
check_q4(reviews.price-med)
# Your code here
reviews.loc[(reviews.points / reviews.price).argmax()].title
# check_q5(reviews.loc[(reviews.points/reviews.price)].title)
# median_price = reviews.price.median()
# check_q5(reviews.price.apply(lambda v: v - median_price))
# Your code here
# reviews.head()
tropial = reviews['description'].map(lambda l:'tropical'in l).value_counts()[True]
# print(tropial)
fruity = reviews['description'].map(lambda l:'fruity'in l).value_counts()[True]
# print(fruity)
pd.Series(data=[tropial,fruity],index=['tropical',fruity])
# Your code here
df = reviews[['country','variety']]
df.dropna()
s = ((df.country + '-'+df.variety).value_counts())
check_q7(s)