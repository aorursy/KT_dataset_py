import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews['points'].median()
# Your code here
#As there is a nan value in the country column to get the actual countries represented I have dropped nan values and got the unique values
reviews['country'].dropna().unique()
# Your code here
reviews['country'].value_counts()
# Your code here
medianPrice = reviews['price'].median()
reviews['price'].map(lambda v:v-medianPrice)
# Your code here
reviews.loc[(reviews.points/reviews.price).idxmax()].title
# Your code here
tropicalWine=reviews.description.map(lambda a: "tropical" in a).value_counts()
fruityWine=reviews.description.map(lambda f: 'fruity' in f).value_counts()
pd.Series([tropicalWine[True], fruityWine[True]], index =['tropical', 'fruity'])
# Your code here
df = reviews.loc[(reviews['country'].notnull()) & (reviews['variety'].notnull())]
df = df.apply(lambda srs:srs.country + "-" + srs.variety, axis='columns')
df