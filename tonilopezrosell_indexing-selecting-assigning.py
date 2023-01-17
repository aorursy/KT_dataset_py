import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews ['description']

description_column = reviews ['description']
description_column [0]
reviews.loc [0]
first_10_description = description_column [0:10]
s = pd.Series (first_10_description)
s
s = pd.DataFrame (reviews.loc [[1,2,3,4,5,8], :])
s
r = pd.DataFrame (reviews.loc [[0,1,10,100], ['country', 'province','region_1', 'region_2']])
r
f = pd.DataFrame (reviews.loc [:100, ['country', 'variety']])
f
y = pd.DataFrame (reviews.loc[reviews.country == 'Italy'])
y
    
g = pd.DataFrame (reviews.loc [reviews.region_2.notnull()])
g
w = pd.DataFrame (reviews.loc [:,'points'])
w
q = pd.DataFrame (reviews.loc [:999,'points'])
q
v = pd.DataFrame (reviews.loc [reviews.shape [0] -1000:,'points'])
v

z = pd.DataFrame (reviews.loc[reviews.country == 'Italy', ['country','points']])
z
i = pd.DataFrame (reviews.loc[(reviews.points >= 90) & ((reviews.country == 'Italy') | (reviews.country == 'France'))])
i