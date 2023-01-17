import pandas as pd

pd.set_option('max_rows', 5)

import numpy as np

from learntools.advanced_pandas.summary_functions_maps import *



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())

# reviews['points'].median()

#the median value of 'points' is 88.0
check_q2(reviews.country.unique())

# print(answer_q2())

# len(reviews.country.unique())#:44

# type(reviews.country.unique())#:numpy.ndarray

# check_q2(reviews.country.unique)#:missing error '()'

# type(reviews.country)#:pandas.core.series.Series

check_q3(reviews.country.value_counts())

# reviews.country.describe()#:output overview of the column 'country',also dispaly the 'US' is the most often

# print(answer_q3())

# reviews.country.value_counts().head()
#check_q4(reviews.price.map(lambda p: p - reviews.price.median()))#error:cpu100% and no output

# print(answer_q4())

#solution1:

median_price = reviews.price.median()

check_q4(reviews.price.map(lambda v: v - median_price))


# check_q5(reviews.price.apply(lambda v: v - median_price))#this solution is same as q4,may be not reasonable

#?"how to solve the points-to-price ratio in the dataset" 

#note:may be the order of questions does not match with the order of answers

check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()])#this is a reasonable solution for Exerise 5
#the solution of Exercise 6

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()

fruit_wine = reviews.description.map(lambda r: "fruit" in r).value_counts()

pd.Series([tropical_wine[True],fruit_wine[True]],index=['tropical','fruit'])
#the first solution from the tutorial 

ans = reviews.loc[(reviews.country.notnull())&(reviews.variety.notnull())]

ans = ans.apply(lambda srs: srs.country + "-" + srs.variety,axis = 'columns')

ans.value_counts()

# the second solution from shawn

def contact(row):

    row_contact = row.country + "-" + row.variety

    return row_contact

ans = reviews.loc[(reviews.country.notnull())&(reviews.variety.notnull())]

ans = ans.apply(contact,axis='columns')

ans.value_counts()