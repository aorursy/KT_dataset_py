import pandas as pd

import seaborn as sns

from learntools.advanced_pandas.indexing_selecting_assigning import *



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
check_q1(reviews.description)

#reviews['description'],also right
check_q2(reviews['description'][0])



#also right,reviews.description[0]
check_q3(reviews.iloc[0])
check_q4(reviews.iloc[0:10,1])
print(answer_q4())
check_q5(reviews.iloc[[1,2,3,5,8]])
check_q6(reviews.iloc[[0,1,10,100],[0,5,6,7]])

#reviews.iloc[[0,1,10,100],[0,5,6,7]]

#reviews.iloc[[0,1,10,100],["country","province","region_1","region_2"]] is error 
check_q7(reviews.loc[0:99,['country','variety']])

#check_q7(reviews.loc(0:99,['country','variety']))
print(answer_q7())
check_q8(reviews.loc[reviews["country"] == 'Italy'])

#check_q8(reviews.loc[:,reviews.loc['country']=='Italy'])#error syntax

#reviews.loc[:,reviews.loc[1] != 0]
print(answer_q8())
check_q9(reviews.loc[reviews['region_2'].notnull()])

# print(answer_q9())
check_q10(reviews['points'])

# print(answer_q10())
check_q11(reviews.points[:999])#also is right

# type(reviews.loc[:999,'points'])

# check_q11(reviews.loc[:999,['points']]) #is wrong

# check_q11(reviews.loc[:999,['points']])



# print(answer_q11())
check_q12(reviews.loc[128971:129971,'points'])#is also right

# (reviews.loc[-1:,'points'])#is wrong

# reviews.shape

# check_q12(reviews.points[-1000:-1])#is right

# print(answer_q12())

# (reviews.iloc[-1000:,3]).shape#is right
# check_q13(reviews.iloc[reviews['country']=='Italy','points'])#is wrong

# check_q13(reviews.loc[reviews['country'] == 'Italy','points'])#is right

check_q13(reviews[reviews.country == 'Italy'].points)# is also right



# print(answer_q13())
# check_q14(reviews.loc[reviews.points >= 90,reviews.country == 'Italy'or'France'])#wrong

# print(answer_q14())

# reviews[reviews.country.isin(['Italy','France'])&(reviews.points >=90)].country#is also right

check_q14(reviews.loc[reviews.country.isin(['Italy','France'])&(reviews.points>=90),'country'])#is also right