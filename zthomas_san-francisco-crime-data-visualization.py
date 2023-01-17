import numpy as np 

import pandas as pd 

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import sklearn
train = pd.read_csv('../input/train.csv')

train.head()
train.columns = [col.lower() for col in train.columns]



train.columns
train.category.value_counts().plot(kind='barh', figsize=(10,8))
pd.pivot_table(train, index = 'pddistrict', columns = 'category', aggfunc=len)



#Useful, but kind of not. Let's see if we can visualize ...
train.dates = pd.to_datetime(train.dates)

train.dtypes
train.dates.apply(lambda x: x.hour).value_counts().sort_index().plot(kind='line')
train.dayofweek.value_counts()[['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']]#.plot(kind='line')