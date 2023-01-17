import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
check_q1(reviews['description'])
answer_q1()
# Your code here
check_q2(reviews.description[0])
# Your code here
check_q3(reviews.iloc[0])
answer_q3()
# Your code here
ten_values = reviews.description[:10]
check_q4(ten_values)
answer_q4()
# Your code here
df = reviews.iloc[[1,2,3,5,8], :]
check_q5(df)
answer_q5()
# Your code here
check_q6(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']])
# Your code here
check_q7(reviews.loc[0:100, ['country', 'variety']])
# Your code here
check_q8(reviews.loc[reviews.country=='Italy'] )
# Your code here
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.points[:1000])
# Your code here
check_q12(reviews.points[-1000:])
answer_q13()
# Your code here
check_q13(reviews[reviews.country=='Italy'].points)
# Your code here
check_q14(reviews[reviews.country.isin(['France', 'Italy']) & (reviews.points >= 90)].country)