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
check_q1(reviews.description)
# Your code here
check_q2(reviews['description'][0])
# Your code here
fr = reviews.iloc[0,:]
check_q3(fr)
# Your code here
ft = reviews['description'][0:10]
check_q4(ft)
# Your code here
sc = reviews.iloc[[1,2,3,5,8],:]
check_q5(sc)
# Your code here
sc = reviews.iloc[[0,1,10,100],[0,5,6,7]]
check_q6(sc)
# Your code here
sc = reviews.loc[0:100,['country','variety']]
#sc1 = reviews.iloc[0:101,[0,11]]
check_q7(sc)
# Your code here
wi = reviews.loc[reviews.country == 'Italy']
check_q8(wi)
# Your code here
wnn = reviews.loc[reviews.region_2.notnull()]
check_q9(wnn)
# Your code here
wp = reviews.points
check_q10(wp)
#check_q10(reviews.points)
# Your code here
wp = reviews.points[0:1000]
check_q11(wp)
# Your code here
wpl = reviews.points[-1000:]
check_q12(wpl)
# Your code here
wip = reviews.points[reviews.country == 'Italy']
check_q13(wip)
# Your code here
wfip = reviews.country[((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90)]
check_q14(wfip)
