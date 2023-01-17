import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
n = reviews.points.median()
check_q1(n)
un = reviews.country.unique()
check_q2(un)
ki = reviews.country.value_counts()
check_q3(ki)
price_median = reviews.price.median()
dd = reviews.price.map(lambda p: p - price_median)
check_q4(dd)
def remed_points(srs):
    srs.points = srs.points - price_median
    return srs
reviews.apply(remed_points, axis='columns')
def ptp(srs):
    srs.ptp = srs.points/srs.price
    return srs
reviews.apply(ptp,axis='columns')
reviews.iloc[(reviews.points/reviews.price).argmax()].title
reviews.description.head().T
trop = reviews.description.map(lambda t: "tropical" in t).value_counts()
fruit = reviews.description.map(lambda f: 'fruity' in f).value_counts()
ans = pd.Series([trop[True], fruit[True]], index=['Tropical','Fruity'])
ans
DS = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
countype = pd.Series((DS.country+' - '+DS.variety).value_counts())
countype