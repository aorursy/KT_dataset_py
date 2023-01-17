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
mprice=reviews.price.median()
reviews.price.map(lambda p: p-mprice)
answer_q4()
mprice = reviews.price.median()
reviews.apply(lambda v: v.price-mprice,axis=1)

mprice = reviews.price.median()

def newprice(df):
    df.price = df.price - mprice
    return df
#newprice(reviews)
reviews.apply(newprice, axis=1)



reviews
reviews.loc[(reviews.points/reviews.price).idxmax(),'title']

answer_q6()
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

tropical = reviews.description.loc[reviews.description.str.contains('tropical')].count()
fruity = reviews.description.loc[reviews.description.str.contains('fruity')].count()
word_count = pd.Series([tropical, fruity], index=['tropical', 'fruity'])
word_count
#edit 2018-06-03
reviews.loc[(reviews.variety.notna()) & (reviews.country.notna())].apply(lambda x: x.country + '-' + x.variety, axis=1).value_counts()
newreviews = reviews[(reviews.country.notna()) & (reviews.variety.notna())]
combo = newreviews.apply(lambda v: v.country + '-' + v.variety, axis=1)
combo.value_counts()