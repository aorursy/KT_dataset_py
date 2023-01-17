import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
med = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - med))
# Your code here
check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
tropical = reviews.description.map(lambda r: 'tropical' in r).value_counts()
fruity = reviews.description.map(lambda r: 'fruity' in r).value_counts()
check_q6(pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity']))
# Your code here
df = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
df = df.apply(lambda el: el.country + " - " + el.variety, axis = 1)
check_q7(df.value_counts())