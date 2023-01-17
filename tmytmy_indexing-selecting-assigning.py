import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews.description)
check_q2(reviews.description[0])
check_q3(reviews.iloc[0])
check_q4(reviews.iloc[0:10,1])
check_q5(reviews.iloc[[1,2,3,5,8]])
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
check_q7(reviews.loc[0:100,['country','variety']])
#check_q8(reviews[reviews['country']=='Italy'])
check_q8(reviews[reviews.country=='Italy'])
check_q9(reviews.loc[reviews.region_2.notnull()])
check_q10(reviews.points)
check_q11(reviews.points.iloc[0:1001])
check_q12(reviews.iloc[-1000:,3])
abc = reviews.loc[reviews.country=='Italy','points']
abc
check_q13(abc)
#reviews[(reviews['country']=='Italy')| (reviews['country']=='France') & (reviews['points']>90)]
ans =reviews[reviews.country.isin(['France','Italy']) & (reviews.points >=90)].country
check_q14(ans)