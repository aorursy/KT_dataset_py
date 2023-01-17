import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
data = reviews['description']
data
check_q1(data)
data = reviews['description'].iloc[0,]
check_q2(data)
data = reviews.iloc[0,]
check_q3(data)
data = reviews['description'].iloc[0:10,]
check_q4(data)
data = reviews.iloc[[1,2,3,5,8],]
check_q5(data)
data = reviews[['country', 'province', 'region_1','region_2']].iloc[[0,1,10,100],]
check_q6(data)
data = reviews[['country','variety']].iloc[0:101,]
check_q7(data)
reviews.head()
data = reviews[reviews.country == 'Italy']
data
check_q8(data)
data = reviews[reviews['region_2'].notnull()]
data
check_q9(data)
data = reviews['points']
check_q10(data)
data = reviews['points'].iloc[0:1001,]
data
check_q11(data)
data = reviews['points'].iloc[-1000:,]
data
check_q12(data)
data = reviews['points'][reviews.country == 'Italy']
data
check_q13(data)
reviews.head()
data = reviews['country'][((reviews.country == 'France') | (reviews.country == 'Italy')) & (reviews.points >=90)]
data
check_q14(data)