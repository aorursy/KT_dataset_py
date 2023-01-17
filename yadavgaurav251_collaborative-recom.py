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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from surprise import Reader,Dataset,SVD
from surprise.accuracy import rmse,mae
from surprise.model_selection import cross_validate
print("import success")

movies=pd.read_csv("/kaggle/input/newdataset/movies.csv")
ratings=pd.read_csv("/kaggle/input/newdataset/ratings.csv")
ratings.head()
ratings.drop('timestamp',axis=1,inplace=True)
ratings.head()
ratings.isna().sum()
n_movies=ratings['movieId'].nunique()
n_users=ratings['userId'].nunique()
print(f"number of unique movies: {n_movies}")
print(f'number of unique users: {n_users}')
available_ratings=ratings['rating'].count()
total_ratings = n_movies* n_users
missing_ratings = total_ratings - available_ratings
sparsity=(missing_ratings/total_ratings)*100
print(f'Sparsity = {sparsity}')
ratings['rating'].value_counts().plot(kind='bar')
filter_users = ratings['userId'].value_counts() > 3
filter_users = filter_users[filter_users].index.tolist()
print(filter_users)
filter_movies = ratings['movieId'].value_counts() > 3
filter_movies = filter_movies[filter_movies].index.tolist()
print(filter_movies)
print("Original -")
print(ratings.shape)
ratings =ratings[(ratings['movieId'].isin(filter_movies)) & (ratings['userId'].isin(filter_users))]
print("New shape -")
print(ratings.shape)
cols = ['userId','movieId','rating']
reader = Reader(rating_scale = (0.5,5))
data = Dataset.load_from_df(ratings[cols],reader)
trainset = data.build_full_trainset()
antiset = trainset.build_anti_testset()
algo = SVD(n_epochs = 25 ,verbose =True)
cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True)
print("Training Completed , Yeahhh !!! ")
predictions= algo.test(antiset)

predictions[0]
from collections import defaultdict
def get_top_n(predictions,n):
    top_n= defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid,est))
                          
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x:x[1],reverse= True)
        top_n[uid]=user_ratings[:n]
    return top_n
    pass
top_n = get_top_n(predictions,n=3)
for uid, user_ratings in top_n.items():
    print("UserId - ",uid,"Recommended MovieID - ",[iid for (iid,rating) in user_ratings])
top_n = get_top_n(predictions,n=3)
for uid, user_ratings in top_n.items():
    print("UserId - ",uid,"Recommended Movies - ",[movies["title"][iid-1] for (iid,rating) in user_ratings])
movies.head()
movies["title"]
print(top_n)
results= pd.DataFrame(top_n)
movies["title"][0]
results.describe()