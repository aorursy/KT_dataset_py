import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
description = reviews['description']

check_q1(description)
# Your code here
first_description = description[0]

check_q2(first_description)
# Your code here
first_row = reviews.iloc[0]

print(first_row)
check_q3(first_row)
# Your code here

first_ten = pd.Series(description.loc[:9])
# 0 to 9 is first ten

check_q4(first_ten)
# Your code here
wine_subset = reviews.iloc[[1,2,3,5,8]]

check_q5(wine_subset)
# Your code here
review_geo = reviews[['country', 'province', 'region_1', 'region_2']]
review_subset = review_geo.iloc[[0, 1, 10, 100]]

check_q6(review_subset)
# Your code here
first_hundred_wines = reviews[['country', 'variety']].loc[0:99]

check_q7(first_hundred_wines)
# Your code here
italian_wines = reviews[ reviews.country == 'Italy' ]

check_q8(italian_wines)
# Your code here
two_regions = reviews[reviews['region_2'].notnull()]

check_q9(two_regions)
# Your code here
points = reviews['points']
check_q10(points)
# Your code here
first_thousand = points.head(1000)

check_q11(first_thousand)
# Your code here
last_thousand = points.tail(1000)

check_q12(last_thousand)
# Your code here
italian_points = italian_wines['points']

check_q13(italian_points)
# Your code here
france_or_italy = reviews[ reviews.country.isin(['Italy', 'France']) ]
with_high_points = france_or_italy[france_or_italy.points >= 90]

check_q14(with_high_points['country'])