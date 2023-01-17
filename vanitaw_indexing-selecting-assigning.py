import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
reviews.description[0]
reviews.loc[0]
reviews.description.iloc[:10]
reviews.iloc[[1,2,3,5,8],:]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
reviews.loc[0:100, ['country','variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
reviews['points']
reviews.points.iloc[0:1000]
reviews.points.iloc[-1000:]
reviews.points.loc[reviews.country=='Italy']
reviews.country.loc[(reviews.country == 'France') & (reviews.points <=90) | (reviews.country == 'Italy') & (reviews.points <=90)]