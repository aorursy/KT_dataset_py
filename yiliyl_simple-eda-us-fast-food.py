# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../input/FastFoodRestaurants.csv')
df.head(3).T
df['province'] = df['province'].astype('category').cat.codes
ax = df.plot(kind="scatter", x="longitude", y="latitude", c="province", cmap=plt.get_cmap("jet"),
        label='Fast Food Restaurants', title='Fast Food Restaurants Location Map', figsize=(20,8))
ax.set_xlabel("longitude")
df.name.value_counts()[:10]