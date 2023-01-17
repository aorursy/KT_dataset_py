import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from surprise import Dataset 

from surprise import Reader

from surprise.model_selection import train_test_split

from surprise import accuracy

from surprise.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score,classification_report
df = pd.read_csv("/kaggle/input/the-movies-dataset/ratings_small.csv")

df.head()
df.isna().sum()
dup_bool = df.duplicated(['userId','movieId','rating'])

print("Number of duplicate records:",sum(dup_bool))
print("Total no of ratings :",df.shape[0])

print("No. of unique users:", df["userId"].nunique())

print("No. of unique movies:", df["movieId"].nunique())
fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('Ratings distribution', fontsize=15)

sns.countplot(df['rating'])

ax.set_xlabel("ratings in interval")

ax.set_ylabel("Total number of ratings")
ratings_per_user = df.groupby(by='userId')['rating'].count()#.sort_values(ascending=False)

ratings_per_user.describe()
ratings_per_movie = df.groupby(by='movieId')['rating'].count()

ratings_per_movie.describe()
reader = Reader()

ratings = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
train_ratings, test_ratings = train_test_split(ratings, test_size=.20, random_state = 42)

print("Size of trainset: ", train_ratings.n_ratings)

print("Size of testset: ", len(test_ratings))
from surprise import BaselineOnly
baseline_model = BaselineOnly(verbose = False)

baseline_model.fit(train_ratings)
train_predictions = baseline_model.test(train_ratings.build_testset())

test_predictions = baseline_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions,verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions,verbose = False))
movies = pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv")

movies.head()
def get_top_n_recommendations(userId,predictions, n=5):

    predict_ratings = {}

    # loop for getting predictions for the user

    for uid, iid, true_r, est, _ in predictions:

        if (uid==userId):

            predict_ratings[iid] = est

    predict_ratings = sorted(predict_ratings.items(), key=lambda kv: kv[1],reverse=True)[:n]

    top_movies = [i[0] for i in predict_ratings]

    top_movies = [str(i) for i in top_movies]

    print("="*10,"Recommended movies for user {} :".format(userId),"="*10)

    print(movies[movies["id"].isin(top_movies)]["original_title"].to_string(index=False))

get_top_n_recommendations(450,test_predictions)
from surprise import KNNBasic

knn_model = KNNBasic(random_state = 42,verbose = False)

knn_model.fit(train_ratings)
train_predictions = knn_model.test(train_ratings.build_testset())

test_predictions = knn_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
param_grid = {'k': list(range(10,45,5)),

             'min_k' : list(range(5,11))}

gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], return_train_measures = True, cv = 5)

gs.fit(ratings)

gs.best_params['rmse']
tuned_knn_model = KNNBasic(k = 15, min_k= 5,random_state = 42, verbose = False)

tuned_knn_model.fit(train_ratings)

train_predictions = tuned_knn_model.test(train_ratings.build_testset())

test_predictions = tuned_knn_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
knn_model_item_based = KNNBasic(user_based = False, random_state = 42)

knn_model_item_based.fit(train_ratings)
train_predictions = knn_model_item_based.test(train_ratings.build_testset())

test_predictions = knn_model_item_based.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
from surprise import SVD
svd_model = SVD(random_state = 42)

svd_model.fit(train_ratings)
train_predictions = svd_model.test(train_ratings.build_testset())

test_predictions = svd_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
param_distributions  = {'n_factors': list(range(50,160,10)),'reg_all': np.arange(0.02,0.2,0.02),'n_epochs' : list(range(1,51))}

rs = RandomizedSearchCV(SVD, param_distributions, measures=['rmse'], return_train_measures = True, cv = 5, n_iter = 20)

rs.fit(ratings)

rs.best_params['rmse']
tuned_svd_model = SVD(n_factors=130, reg_all =0.1, n_epochs = 50, random_state = 42,verbose = False)

tuned_svd_model.fit(train_ratings)

train_predictions = tuned_svd_model.test(train_ratings.build_testset())

test_predictions = tuned_svd_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
from surprise import NMF
nmf_model = NMF(random_state = 42)

nmf_model.fit(train_ratings)
train_predictions = nmf_model.test(train_ratings.build_testset())

test_predictions = nmf_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
param_distributions  = {'n_factors': list(range(10,20,2)),'reg_pu': np.arange(0.02,0.2,0.02),'reg_qi': np.arange(0.02,0.2,0.02), 'n_epochs' : list(range(1,51))}

rs = RandomizedSearchCV(NMF, param_distributions, measures=['rmse'], return_train_measures = True, cv = 5, n_iter = 20)

rs.fit(ratings)

rs.best_params['rmse']
#tuned_nmf_model = NMF(n_factors=18, reg_pu = 0.06, reg_qi = 0.16, n_epochs = 38, random_state = 42)

tuned_nmf_model = NMF(n_factors=18, reg_pu = 0.13999999999999999, reg_qi = 0.12000000000000001, n_epochs = 34, random_state = 42)

tuned_nmf_model.fit(train_ratings)

train_predictions = tuned_nmf_model.test(train_ratings.build_testset())

test_predictions = tuned_nmf_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions, verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose = False))
userIds = []

movieIds = []

ratings = []

for (uid, iid, rating) in train_ratings.all_ratings():

    userIds.append(train_ratings.to_raw_uid(uid))

    movieIds.append(train_ratings.to_raw_iid(iid))

    ratings.append(rating)

dict = {'userId': userIds, 'movieId': movieIds, 'rating': ratings}

training_df = pd.DataFrame(dict)
user_averages = training_df.groupby("userId")["rating"].mean()

user_averages
train_actual_labels = []

train_predicted_labels = []

for uid, iid, r_ui, est, _ in train_predictions:

    if((r_ui - user_averages[uid])>0):

        train_actual_labels.append("Yes")

    else:

        train_actual_labels.append("No")

    if((est - user_averages[uid])>0):

        train_predicted_labels.append("Yes")

    else:

        train_predicted_labels.append("No")
print("Training data distribution of liked movies derived from actual ratings")

print(pd.Series(train_actual_labels).value_counts())

print("\nTraining data distribution of liked movies derived from predicted ratings")

print(pd.Series(train_predicted_labels).value_counts())
test_actual_labels = []

test_predicted_labels = []

for uid, iid, r_ui, est, _ in test_predictions:

    if((r_ui - user_averages[uid])>0):

        test_actual_labels.append("Yes")

    else:

        test_actual_labels.append("No")

    if((est - user_averages[uid])>0):

        test_predicted_labels.append("Yes")

    else:

        test_predicted_labels.append("No")
print("Test data distribution of liked movies derived from predicted ratings")

print(pd.Series(test_actual_labels).value_counts())

print("\nTest data distribution of liked movies derived from predicted ratings")

print(pd.Series(test_predicted_labels).value_counts())
print("Confusion matrix on test data")

confusion_matrix(test_actual_labels,test_predicted_labels)
print("Training data precision : ", precision_score(train_actual_labels,train_predicted_labels,pos_label="Yes"))

print("Test data precision : ", precision_score(test_actual_labels,test_predicted_labels,pos_label="Yes"))
print("Training data recall : ", recall_score(train_actual_labels,train_predicted_labels,pos_label="Yes"))

print("Test data recall : ", recall_score(test_actual_labels,test_predicted_labels,pos_label="Yes"))
print("="*20, "Classification Report", "="*20)

print(classification_report(test_actual_labels,test_predicted_labels))
print(tuned_nmf_model.predict(672, 1721)) # unknown user id but known movie id

print(tuned_nmf_model.predict(43, 2277))  # known user id but unknown movie id

print(tuned_nmf_model.predict(671, 2277)) # unknown user id and unknown movie id
biased_nmf_model = NMF(biased = True,random_state = 42)

biased_nmf_model.fit(train_ratings)

train_predictions = biased_nmf_model.test(train_ratings.build_testset())

test_predictions = biased_nmf_model.test(test_ratings)

print("RMSE on training data : ", accuracy.rmse(train_predictions,verbose = False))

print("RMSE on test data: ", accuracy.rmse(test_predictions,verbose = False))
print(biased_nmf_model.predict(672, 1721)) # unknown user id but known movie id

print(biased_nmf_model.predict(43, 2277))  # known user id but unknown movie id

print(biased_nmf_model.predict(671, 2277)) # unknown user id and unknown movie id