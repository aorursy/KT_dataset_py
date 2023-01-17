import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
reviews["points"].median()
answer_q1()
reviews["country"].unique()
reviews["country"].value_counts()
reviews["price"]=reviews["price"]-reviews["price"].median()
reviews["price"]
pm=reviews['price'].median()
reviews['price'].map(lambda p: p - pm)
import numpy as np
reviews['ptop_ratio']=reviews['points']/reviews['price']
reviews['title'][reviews[np.isfinite(reviews['ptop_ratio'])]['ptop_ratio'].idxmax()]   
#reviews['ptop_ratio'][62]
tropical=reviews['description'].map(lambda x:'tropical'in x).value_counts()[True]
fruity=reviews['description'].map(lambda x: 'fruity'in x).value_counts()[True]
pd.Series([tropical, fruity], index=['Tropical', 'Fruity'])
reviews['country-variety']=reviews['country']+"-"+reviews['variety']
reviews['country-variety'].dropna().value_counts()