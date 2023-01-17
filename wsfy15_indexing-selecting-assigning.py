import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
desc = reviews['description']
print(desc.head())
check_q1(desc)
# Your code here
print(desc[0])

check_q2(desc[0])
# Your code here
df3 = reviews.ix[0]
print(df3)
check_q3(df3)
print(answer_q3())
# Your code here
df4 = reviews.iloc[0:10, 1]
print(df4)
check_q4(df4)
# Your code here
df5 = reviews.iloc[[1, 2, 3, 5, 8]]
print(df5)
check_q5(df5)
# Your code here
df6 = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
print(df6)
check_q6(df6)
# Your code here
df7 = reviews.ix[:100, ['country', 'variety']]
print(df7.head())
check_q7(df7)
# Your code here
df8 = reviews[reviews['country']=='Italy']
print(df8.head())
check_q8(df8)
# Your code here
df9 = reviews[~(reviews['region_2'].isnull())]
print(df9.head())
check_q9(df9)
print(answer_q9())
# Your code here
df10 = reviews.points
check_q10(df10)
# Your code here
df11 = reviews[:1000].points
check_q11(df11)
# Your code here
df12 = reviews[-1000:].points
check_q12(df12)
# Your code here
df13 = reviews.loc[reviews['country']=='Italy']
df13 = df13.points
check_q13(df13)
# Your code here
df14 = reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)].country
print(df14.head())
check_q14(df14)
#print(answer_q14())