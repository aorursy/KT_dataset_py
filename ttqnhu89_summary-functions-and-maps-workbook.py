import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
print(answer_q4())
med = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - med))
print(answer_q5())
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
print(answer_q7())
tropical_list = reviews.description.map(lambda a: 'tropical' in a).value_counts()
fruit_list = reviews.description.map(lambda b: 'fruit' in b).value_counts()
#tropical_list
#fruit_list
pd.Series([tropical_list, fruit_list], index=['tropical','fruity'])
pd
#pd.Series([tropical_list[True], fruit_list[False]], index=['tropical','fruity'])
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()
data = pd.DataFrame(reviews.loc[reviews.country.notnull() & reviews.variety.notnull()])
data_new = pd.Series([i for i in map(lambda x,y:x +' - '+y, data.country, data.variety)])
check_q8(data_new.value_counts())