# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# we must import data after import libraries

#lets do it

movie=pd.read_csv("../input/movielens-20m-dataset/movie.csv")

rating = pd.read_csv("../input/movielens-20m-dataset/rating.csv")
#now it's time we get the know our data

movie.head(10)
#we don't need "genres" feature so we drop that column

movie=movie.drop(["genres"],axis=1)
rating = rating.loc[:,["userId","movieId","rating"]]

rating.head(10)

# now we have an idea about the data

# lets processing a little more

data = pd.merge(movie,rating)
# now lets look at our data 

data.head(10)
#its looking ready to process almost.

#but we have so important problem.

#data is too big.

data.shape
#data has 20 milions row

#we can't not use all and already we don't need it

#this kernel was written for learning purposes 

data = data.iloc[:1000000,:]

#one millions sample is enough for we 

#lets make a pivot table in order to make rows are users and columns are movies.

pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")

pivot_table.head(10)
# now it's time to get our results

# I want to try for "Toy Story"

movie_watched = pivot_table["Toy Story (1995)"]

similarity_with_other_movies = pivot_table.corrwith(movie_watched) 

similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)

similarity_with_other_movies.head()