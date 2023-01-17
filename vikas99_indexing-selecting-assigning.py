import pandas as pd

import seaborn as sns

from learntools.advanced_pandas.indexing_selecting_assigning import *



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])
check_q2(reviews['description'][0])
check_q3(reviews.iloc[0,:])
check_q4(reviews['description'].head(10))
check_q5(reviews.iloc[[1,2,3,5,8],:])
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
check_q7(reviews.loc[0:99,['country','variety']])
check_q8(reviews[reviews['country']=='Italy'])
check_q9(reviews[reviews['region_2'].notnull()])
check_q10(reviews['points'])
reviews.points.head(1000)
reviews['points'].tail(1000)
reviews[reviews['country']=='Italy']['points']
reviews[(reviews['points']>=90) & ((reviews['country']=='Italy') | (reviews['country']=='France'))]['country']