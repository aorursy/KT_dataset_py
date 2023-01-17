# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# read data from wine reviews datase
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0, header=0)
reviews.head(5)
_ = reviews['country'].value_counts().head(10).plot.bar()
# 若想查看各国家的葡萄酒reviews占比，传递 normalized=True给value_counts
_ = reviews['country'].value_counts(normalize=True).head(10).plot.bar(grid=True)
_ = reviews.points.value_counts().sort_index().plot.bar(grid=True)
# line plot
_ = reviews.points.value_counts().sort_index().plot.line(color='green')
# area plot
_ = reviews.points.value_counts().sort_index().plot.area(color='green')
reviews.price.describe(percentiles=[.01,.99])
_ = reviews[ reviews.price < 160].price.plot.hist(bins=20)
# read pokemon dataset
pd.set_option('max_columns', None)
pokemon = pd.read_csv('../input/pokemon/pokemon.csv')
pokemon.head(3)
pokemon.shape
pokemon['type1'].value_counts().plot.bar()
pokemon.hp.value_counts().sort_index().plot.line()
pokemon.weight_kg.plot.hist