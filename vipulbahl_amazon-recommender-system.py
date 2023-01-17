
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import split
from surprise import Dataset,Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from collections import defaultdict

%matplotlib inline
sns.set(style="darkgrid",color_codes=True)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# load the dataset
rdata = pd.read_csv('/kaggle/input/amazon-product-reviews/ratings_Electronics (1).csv',names=['userid','productid','rating','timestamp'])
# lets make a copy of the data so that all the transformation is done on the copy and not on the main dataset
t1data=rdata.copy()
# step 2.1: browse through the first few columns
t1data
## identifying the range of the ratings
np.sort(t1data['rating'].unique())
### analysing the first few records 
t1data.head()
#dropping timestamp column since it is not much of value add
t1data=t1data.drop("timestamp",axis=1)
t1data.head()
t1data.info()
## count of each attribute in the dataset

unique_users =len(np.unique(t1data.userid))
unique_pdts = len(np.unique(t1data.productid))
print('Total number of users is: ',unique_users,'\n')
print('Total number of products is: ',unique_pdts,'\n')
### lets analyse the spread of data
t1data.describe().T
# Identify Duplicate records in the data 
# It is very important to check and remove data duplicates. 
# Else our model may break or report overly optimistic / pessimistic performance results
dupes=t1data.duplicated()
print(' The number of duplicates in the dataset are:',sum(dupes), '\n','There are no duplicates in the dataset')
# checking if there are any null values
t1data.isnull().any()
a=t1data.groupby('rating')['rating'].count()
# Attributes in the Group
Atr1g1='userid'
Atr2g1='productid'
Atr3g1='rating'
data=t1data
##EDA: Spread
# fig, ax = plt.subplots(1,2,figsize=(16,8)) 
plt.figure(figsize=(8,6))
sns.distplot(data[Atr3g1]);
# EDA: count of ratings:
plt.figure(figsize=(8,6))
sns.countplot(data[Atr3g1]);
t2data=t1data.copy()
t2data = t2data[t2data.groupby('userid')['userid'].transform('size') > 49]
t2data=pd.DataFrame(t2data)
t2data=t2data.reset_index(drop=True)
t2data.head()
shape_t2data=t2data.shape
print('The shape of the new dataframe is',shape_t2data,'which means there are',shape_t2data[0],'rows of ratings and',shape_t2data[1],'attributes of userid, productid and rating.')
## lets check the count of ratings given by the users
ratings_per_user = t2data.groupby(by='userid')['rating'].count().sort_values(ascending=False)
ratings_per_user
reader = Reader(rating_scale=(1, 5))
t3data=Dataset.load_from_df(t2data[['userid','productid','rating']],reader)
t3data
trainset, testset = train_test_split(t3data, test_size=.30, random_state=1)
print(type(testset))
print(type(trainset))
# First we will group by product ids and then display mean ratings for the products. For better visualization we will display first 5 records.
t2data.groupby('productid')['rating'].mean().head()
## Next we want to look at which product has got the highest rating. FOr the same same we will sort the productid by the mean ratings.
## We then displayed top 10 products which have the highest ratings
# this analysis is also inconclusive since top ratings dont add value without the count
t2data.groupby('productid')['rating'].mean().sort_values(ascending=False).head(10)
## Next lets try and analyse the products which have been rated the most
t2data.groupby('productid')['rating'].count().sort_values(ascending=False).head()
t2data_product_ratings =pd.DataFrame(t2data.groupby('productid')['rating'].mean())
t2data_product_ratings['ratings_count'] = pd.DataFrame(t2data.groupby('productid')['rating'].count())
t2data_product_ratings.head()
t2data_product_ratings.sort_values(by='ratings_count',ascending=False)
t2data_product_ratings['score'] = t2data_product_ratings['rating']*t2data_product_ratings['ratings_count']
plt.figure(figsize=(8,6))
sns.jointplot(x='rating', y='ratings_count', data=t2data_product_ratings, alpha=0.4)
t2data_product_ratings.sort_values(by='score',ascending=False)
print('the top 5 recommendations are:') 
t2data_product_ratings.sort_values(by='score',ascending=False).head()
### Lets build the model
data = t3data
algo_knn = KNNWithMeans()
algo_knn.fit(trainset)
predictions_knn = algo_knn.test(testset)
# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(predictions_knn, verbose=True)
## We could use item-item based collaborative filtering. Since everytime we used it, google colab crashed giving out of memory issues. 
#We tried executing on local machine as well but no luck. Another option could be trucating the data to reduce memory requirements.
# But that approach dint appear apt to follow.

sim_options = {
    "name": ["msd", "cosine","pearson_baseline"],
    "min_support": [3, 4, 5],
    "user_based": [True],
    "k":[5,10,20,30,40,50,100]
    
}
param_grid = {"sim_options": sim_options,"verbose":[True,False]}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"],cv=3)
gs.fit(data)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
algo = KNNWithMeans(sim_options={'name': 'pearson_baseline', 'min_support': 5, 'user_based': True,'k':5},verbose= True,c=3)
algo.fit(trainset)
# run the trained model against the testset
predictions = algo.test(testset)
predictions
print('the top 5 recommendations are:') 
t2data_product_ratings.sort_values(by='score',ascending=False).head()
# get RMSE
print('For the User-based Model, the accuracy of the Test Set is:')
accuracy.rmse(predictions, verbose=True)
cross_validate(algo, t3data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
def get_top_n(predictions,n=5):
  top_n=defaultdict(list)
  for uid,iid,true_r,est,_ in predictions:
    top_n[uid].append((iid,est))
  for uid,user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1],reverse=True)
    top_n[uid]=user_ratings[:n]
  return top_n
top_n=get_top_n(predictions)
print('top 5 recommended products for each user are:')
top_n