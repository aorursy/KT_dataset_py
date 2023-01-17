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
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re
import seaborn as sns
import surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
Data = pd.read_csv('../input/amazon-electronics-rating-datasetrecommendation/ratings_Electronics.csv')
# Take the 10% Data as the Data is Huge and was cluttering the memory
Data =  Data.iloc[:782448]
#read the Dataset
Data = pd.read_csv('../input/amazon-electronics-rating-datasetrecommendation/ratings_Electronics.csv',names=['userId', 'productId','Rating','timestamp'])
Data.dtypes
Data.head()
Data.shape
# Take the 10% Data as the Data is Huge and was cluttering the memory
Data =  Data.iloc[:782448]
# Drop the Timestamp feature as it doesnt add much value to the model

Data = Data.drop(['timestamp'], axis = 1)
Data.shape
# find minimum and maximum ratings
print('The minimum rating is: %d' %(Data['Rating'].min()))
print('The maximum rating is: %d' %(Data['Rating'].max()))
Data.groupby('userId')['Rating'].mean().sort_values(ascending=False).head(10)  
# check the Rating distribution in the range 1-5 for the Data given 

with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=Data, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
print("Total data ")
print("*"*50)
print("\nTotal no of ratings :",Data.shape[0])
print("Total No of Users   :", len(np.unique(Data.userId)))
print("Total No of products  :", len(np.unique(Data.productId)))
#Keep the users where the user has rated more than 50 

counts1 = Data['userId'].value_counts()
#print(counts1)
Data_new = Data[Data['userId'].isin(counts1[counts1 >= 50].index)]
#counts1
#highest rated products from the selected records. 

Data_new.groupby('productId')['Rating'].mean().sort_values(ascending=False) 
#Calculate the density of the rating matrix

final_ratings_matrix = Data_new.pivot(index = 'userId', columns ='productId', values = 'Rating').fillna(0)
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))
final_ratings_matrix.head()
# Matrix with one row per 'Product' and one column per 'user' for Item-based CF
final_ratings_matrix_T = final_ratings_matrix.transpose()
final_ratings_matrix_T.head()
#Count of user_id for each unique product as recommendation score 
Data_new_grouped = Data_new.groupby('productId').agg({'userId': 'count'}).reset_index()
Data_new_grouped.rename(columns = {'userId': 'score'},inplace=True)
Data_new_grouped.head()
#Sort the products on recommendation score 
train_data_sort = Data_new_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5) 
popularity_recommendations 
# Use popularity based recommender model to make predictions
def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations 
find_recom = [15,21,53]   # This list is user choice.
for i in find_recom:
    print("Here is the recommendation for the userId: %d\n" %(i))
    print(recommend(i))    
    print("\n") 
no_of_ratings_per_product = Data_new.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()
# Top 30 recommendations for the users

popular_products = pd.DataFrame(Data_new.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from collections import defaultdict
#Reading the dataset
reader = Reader(rating_scale=(1, 5))
data1 = Dataset.load_from_df(Data_new,reader)
data1
#Splitting the dataset
trainset, testset = train_test_split(data1, test_size=0.3,random_state=10)
trainset.ur
# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)
# run the trained model against the testset
test_pred = algo.test(testset)
test_pred
def get_top_n(predictions, n=5):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
top_n = get_top_n(test_pred, n=5)
top_n
# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
uid = "A11D1KHM7DVOQK"  # raw user id (as in the ratings file). They are **strings**!
iid = "B000059L44"  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=0.0, verbose=True)
pred = pd.DataFrame(test_pred)
pred[pred['uid'] == 'A11D1KHM7DVOQK'][['iid', 'r_ui','est']].sort_values(by = 'est',ascending = False).head(10)
# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
from collections import defaultdict
from surprise import SVD
from surprise import Dataset
data1 = Dataset.load_from_df(Data_new,reader)
data1
trainset, testset = train_test_split(data1, test_size=.15)
trainset.ur
algo = SVD(n_factors=5,biased=False)
algo.fit(trainset)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
testset
predictions = algo.test(testset)
predictions
top_n = get_top_n(predictions, n=5)
top_n
# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
# compute RMSE
accuracy.rmse(predictions)
#User-based Collaborative Filtering
# Matrix with row per 'user' and column per 'item' 
pivot_df = Data_new.pivot(index = 'userId', columns ='productId', values = 'Rating').fillna(0)
pivot_df.shape
pivot_df.head()
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.head()
pivot_df.set_index(['user_index'], inplace=True)

# Actual ratings given by users
pivot_df.head()
from scipy.sparse.linalg import svds
# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df, k = 10)
# Construct diagonal array in SVD
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

# Predicted ratings
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
preds_df.head()
# Recommend the items with the highest predicted ratings

def recommend_items(userID, pivot_df, preds_df, num_recommendations):
      
    user_idx = userID-1 # index starts at 0
    
    # Get and sort the user's ratings
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userID))
    print(temp.head(num_recommendations))
#Enter 'userID' and 'num_recommendations' for the user #
userID = 20
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)
# Predicted ratings 
preds_df.head()
# Average PREDICTED rating for each item
preds_df.mean().head()
rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
print(rmse_df.shape)
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print('\nRMSE SVD Model = {} \n'.format(RMSE))
# Enter 'userID' and 'num_recommendations' for the user #
userID = 20
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)