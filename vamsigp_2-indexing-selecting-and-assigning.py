import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
check_q1(reviews['description'])
# Your code here
check_q2(reviews['description'][0])
# Your code here
check_q3(reviews.iloc[0])
# Your code here
s = reviews['description'][:10]
# type(s)
check_q4(s)
# Your code here
arr = [1,2,3,5,8]
df5 = reviews.iloc[arr]
check_q5(df5)
# Your code here
arr = [0,1,10,100]
pos = [0,5,6,7]
df6 = reviews.iloc[arr,pos]
check_q6(df6)
# Your code hered
df7 = reviews.loc[0:100,['country','variety']]
check_q7(df7)
# Your code here
filter = reviews['country']=='Italy'
df_filter = reviews[filter]
check_q8(df_filter)
# Your code here
filter = reviews.region_2.notnull()
check_q9(reviews[filter])
# Your code here
check_q10(reviews['points'])
# Your code here
check_q11(reviews['points'][:1000])
# Your code here
check_q12((reviews.iloc[-1000:, 3]))
# Your code here
reviews[reviews.country=='Italy']['points']
# Your code here
# filter1 =[(reviews.country=='Italy') | (reviews.country=='France')]# and (reviews.points>90)
fil = reviews['points']>=90
df = (reviews[fil][(reviews.country=='Italy')|(reviews.country=='France')])
check_q14(df['country'])
