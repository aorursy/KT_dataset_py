# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('/kaggle/input/movie-ratings-dataset/movie_ratings.csv').iloc[:,1:]
movies.head(5)

# .iloc[] is used to select specific cells, using the numerical index, which takes two arguments: row index, column index
# .head(n) displays the first n rows of the dataset
movies = movies.set_index('movie')
movies.head(5)

# .set_index() is used to set a column as the index of the dataframe
movies.shape[0]

# .shape gives the size of dataset, with .shape[0] displays rows and .shape[1] displays columns
round(movies.imdb.mean(),2)

# .mean() calculates the mean of selected data
movies.imdb.median()

# .median() calculates the median of selected data
movies.groupby('imdb').size().plot.line()

# .groupby(x) splits data into subgroups based on feature x, i.e. split-apply-combine

# .plot.line() gives the graph, which is a line plot in this case
movies[ movies.imdb >= 9 ].shape[0]

# movies[ movies.imdb >= 9 ] conditional selecting: select all rows with imdb >= 90
movies[ movies.imdb >= 8 ].shape[0]
movies.groupby('year').size()

# .groupby.size() counts frequency - it first splits the dataset into subgroups and then count the frequencies of each subgroup
movies.groupby('year').imdb.mean()

# it first splits the dataset into subgroups and then calculate the mean of each subgroup
movies.groupby('year').imdb.mean().sort_values(ascending=False).head(5)

# .sort_values() is used to sort the data in either ascending or descending order
movies.groupby('year').imdb.mean().plot.line()
movies.sort_values('votes', ascending=False).head(1)
movies.sort_values('imdb', ascending=False).head(1)
movies.sort_values('votes').head(1)
movies.sort_values('imdb').head(1)
adam_sandler = movies.loc[['Mr. Deeds','Anger Management','50 First Dates','The Longest Yard','Click','Reign Over Me','I Now Pronounce You Chuck and Larry','Bedtime Stories','Funny People','Grown Ups','Just Go with It','Jack and Jill','Hotel Transylvania','Grown Ups 2','Blended','Pixels','Hotel Transylvania 2','The Ridiculous']]

# .loc[] is used to select rows using index, not numerical index 0,1,2,etc., but actual index of the dataframe
round(adam_sandler.imdb.mean(),2)