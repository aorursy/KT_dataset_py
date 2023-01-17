# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
df = pd.read_csv('../input/tmdb_5000_movies.csv')
df.head()
# next we want to see some of this data (but not all of it)
df.head()
# so we know there should be exactly 5000 rows of data, but we can check with "shape"

df.shape
# we can see more statistical details about the movie database

df.describe() # only shows numerical columns
df_reduced=df[['budget', 'homepage', 'original_language', 'title', 'popularity', 'release_date', 'revenue', 'runtime']]
df_reduced.head()
df_reduced[df['homepage'].isnull()].head()
# Let's take a look at all of the languages in the dataset

df_reduced['original_language'].unique()
# We can group the data by language

df_reduced.groupby('original_language')['title'].count().sort_values(ascending=False)
# We can get all of the rows for a specific language!

df_reduced[df_reduced['original_language'] == 'zh']
# Let's add a new column to identify movies we like. 

df_reduced['i_like'] = None
df_reduced.head()
# Update the column for a movie you like! 

df_reduced.loc[df_reduced['title'] == 'Avatar', ['i_like']] = True
df_reduced.loc[df_reduced['title'] == 'Avatar']
df_reduced.plot.scatter(x='budget', y='revenue')
ax = df.set_index('budget')['revenue'].plot(style='o', figsize=(12,12))

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
      if(point['x'] > 200000000 or point['y'] > 1000000000):
        ax.text(point['x'], point['y'], str(point['val']))
           
label_point(df.budget, df.revenue, df.title, ax)
df_reduced.plot.scatter(x='runtime', y='revenue')
df_reduced['runtime'].hist(bins=15)
df_reduced.plot.bar(x='original_language', y='revenue')
