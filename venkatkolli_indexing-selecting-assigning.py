import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews ['description']
reviews['description'][0]
reviews.iloc[0]
reviews.iloc[0:10]
reviews.iloc[[1,2,3,5,8]]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]

reviews.loc[reviews.index[0:100],['country','variety']]
reviews[reviews.country=='Italy']
reviews[reviews.region_2.notna()]
reviews['points']
reviews['points'][0:1000]
reviews.loc[reviews.index[-1000:],'points']
reviews['points'][reviews['country']=='Italy']
reviews['country'][reviews.country.isin(['Italy','France']) & (reviews['points']>=90)]