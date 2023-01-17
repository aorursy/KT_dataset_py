import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews.description)
check_q2(reviews['description'][0])
check_q3(reviews.iloc[0])
#print(answer_q4())
reviews.iloc[0:10, 1]
check_q4(reviews.iloc[0:10,1])
check_q5(reviews.loc[(reviews.index == 1) | (reviews.index == 2) | (reviews.index == 3)| (reviews.index == 5) | (reviews.index == 8)])
print(answer_q5())
check_q6(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1','region_2']])
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
check_q7(reviews.loc[:100,['country','variety']])
check_q8(reviews.loc[(reviews.country=='Italy')])
check_q9(reviews.loc[reviews.region_2.notnull()])
print(answer_q9())
check_q10(reviews['points'])
print(answer_q10())
check_q11(reviews['points'][0:1000])
#check_q11(reviews['points'][0:1000])
check_q11(reviews.loc[:1000, 'points'])
print(answer_q11())
check_q12(reviews['points'][-1000:])
check_q12(reviews.iloc[-1000:, 3])
reviews[reviews.country == "Italy"].points
check_q13(reviews.points[reviews.country == 'Italy'])
reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country

check_q14(reviews[((reviews.country == "France") | (reviews.country == 'Italy')) & (reviews.points >= 90)].country)