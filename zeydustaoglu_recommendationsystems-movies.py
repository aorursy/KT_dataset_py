# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
print(os.listdir("../input"))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import movie data set and look at columns
movie = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
movie.columns
# what we need is that movie id and title
movie = movie.loc[:,["movieId","title"]]
movie.head(10)
# import rating data and look at columsn
rating = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")
rating.columns
# what we need is that user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)
# then merge movie and rating data
data = pd.merge(movie,rating)
# now lets look at our data 
data.head(10)
data = data.iloc[:1000000,:]

# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table=pivot_table.fillna(1.0)
pivot_table.head(10)
movie_watched = pivot_table["Ace Ventura: When Nature Calls (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending = False)
similarity_with_other_movies.head()
