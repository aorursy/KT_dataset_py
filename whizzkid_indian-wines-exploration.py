import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/winemag-data-130k-v2.csv",index_col=[0])

data.head(2)
data.country.value_counts()
indian_data=data[data["country"]=='India'].head()
indian_data.head()
indian_data.winery.unique()
indian_data.province.unique()
indian_data.variety.unique()
indian_data.price.value_counts() #vines are cheap in india