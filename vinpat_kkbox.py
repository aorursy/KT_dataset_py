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

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

songs = pd.read_csv('../input/songs.csv')

test = pd.read_csv('../input/test.csv')

members = pd.read_csv('../input/members.csv')

songs_extra_info = pd.read_csv('../input/song_extra_info.csv')
df = train.merge(members,how='inner',on='msno')

df = df.merge(songs,how='inner',on='song_id')

df.head()
!pip install ../input/recommender-system/RecommenderSystems_PyData_2016-master/RecommenderSystems_PyData_2016-master/Evaluation.py

    
df.columns
df = train.merge(members,how='inner',on='msno')

df = df.merge(songs,how='inner',on='song_id')

df.drop_duplicates(subset =['source_system_tab', 'source_screen_name',

       'source_type', 'target', 'city', 'bd', 'gender', 'registered_via',

       'registration_init_time', 'expiration_date', 'song_length', 'genre_ids',

       'artist_name', 'composer', 'lyricist', 'language'], 

                     keep = False, inplace = True) 

df.reset_index(drop = True,inplace=True)

df
sa=df[['msno','song_id','artist_name']]

sa1=df[['msno','song_id']]



sa = sa.groupby(['msno','artist_name']).count()

sa1 = sa1.groupby('msno').count()

sa1







sa.reset_index('artist_name',inplace=True)



sa



final = sa.merge(sa1, left_index=True, right_index=True, how='left')



final['normalized']=final['song_id_x']/final['song_id_y']



final.reset_index('msno',inplace=True)



final = final[['msno','artist_name','normalized']]

final



#final.pivot(index='msno', columns='artist_name', values='normalized')
final = final.set_index('msno')

final
artist_songs = songs[['artist_name','song_id']].groupby(['artist_name']).count().sort_values('song_id', ascending=False)
artist_songs
artist_avg_song_len = (songs[['artist_name','song_length']].groupby(['artist_name']).mean()/1000).sort_values('song_length', ascending=False)
test = songs[['artist_name','song_id','language']].groupby(['artist_name','language']).count()

test.reset_index(level=['artist_name','language'],inplace=True)

test['count_max']=test.groupby(['artist_name'])['song_id'].transform(max)

artist_max_song_language = test.loc[test["song_id"] == test["count_max"]]

artist_max_song_language.reset_index(drop=True)

artist_max_song_language.sort_values('count_max',ascending=False)

test = songs[['artist_name','song_id','genre_ids']].groupby(['artist_name','genre_ids']).count()

test.reset_index(level=['artist_name','genre_ids'],inplace=True)

test['count_max']=test.groupby(['artist_name'])['song_id'].transform(max)

artist_max_song_genre = test.loc[test["song_id"] == test["count_max"]]

artist_max_song_genre.reset_index(drop=True)

artist_max_song_genre.sort_values('count_max',ascending=False)
%matplotlib inline



import pandas

from sklearn.model_selection import train_test_split

import numpy as np

import time

from sklearn.externals import joblib

#from recommenders import Recommenders as Recommenders

#from recommenders import Evaluation as Evaluation

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

#print(check_output(["ls", "C:\\Users\\Vineel\\Documents\\RecommenderSystems_PyData_2016-master\\input"]).decode("utf8"))
#Class to calculate precision and recall



import random



class precision_recall_calculator():

    

    def __init__(self, test_data, train_data, pm, is_model):

        self.test_data = test_data

        self.train_data = train_data

        self.user_test_sample = None

        self.model1 = pm

        self.model2 = is_model

        

        self.ism_training_dict = dict()

        self.pm_training_dict = dict()

        self.test_dict = dict()

    

    #Method to return random percentage of values from a list

    def remove_percentage(self, list_a, percentage):

        k = int(len(list_a) * percentage)

        random.seed(0)

        indicies = random.sample(range(len(list_a)), k)

        new_list = [list_a[i] for i in indicies]

    

        return new_list

    

    #Create a test sample of users for use in calculating precision

    #and recall

    def create_user_test_sample(self, percentage):

        #Find users common between training and test set

        users_test_and_training = list(set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))

        print("Length of user_test_and_training:%d" % len(users_test_and_training))



        #Take only random user_sample of users for evaluations

        self.users_test_sample = self.remove_percentage(users_test_and_training, percentage)



        print("Length of user sample:%d" % len(self.users_test_sample))

        

    #Method to generate recommendations for users in the user test sample

    def get_test_sample_recommendations(self):

        #For these test_sample users, get top 10 recommendations from training set

        #self.ism_training_dict = {}

        #self.pm_training_dict = {}



        #self.test_dict = {}



        for user_id in self.users_test_sample:

            #Get items for user_id from item similarity model

            print("Getting recommendations for user:%s" % user_id)

            user_sim_items = self.model2.recommend(user_id)

            self.ism_training_dict[user_id] = list(user_sim_items["song"])

    

            #Get items for user_id from popularity model

            user_sim_items = self.model1.recommend(user_id)

            self.pm_training_dict[user_id] = list(user_sim_items["song"])

    

            #Get items for user_id from test_data

            test_data_user = self.test_data[self.test_data['user_id'] == user_id]

            self.test_dict[user_id] = set(test_data_user['song'].unique() )

    

    #Method to calculate the precision and recall measures

    def calculate_precision_recall(self):

        #Create cutoff list for precision and recall calculation

        cutoff_list = list(range(1,11))





        #For each distinct cutoff:

        #    1. For each distinct user, calculate precision and recall.

        #    2. Calculate average precision and recall.



        ism_avg_precision_list = []

        ism_avg_recall_list = []

        pm_avg_precision_list = []

        pm_avg_recall_list = []





        num_users_sample = len(self.users_test_sample)

        for N in cutoff_list:

            ism_sum_precision = 0

            ism_sum_recall = 0

            pm_sum_precision = 0

            pm_sum_recall = 0

            ism_avg_precision = 0

            ism_avg_recall = 0

            pm_avg_precision = 0

            pm_avg_recall = 0



            for user_id in self.users_test_sample:

                ism_hitset = self.test_dict[user_id].intersection(set(self.ism_training_dict[user_id][0:N]))

                pm_hitset = self.test_dict[user_id].intersection(set(self.pm_training_dict[user_id][0:N]))

                testset = self.test_dict[user_id]

        

                pm_sum_precision += float(len(pm_hitset))/float(N)

                pm_sum_recall += float(len(pm_hitset))/float(len(testset))



                ism_sum_precision += float(len(ism_hitset))/float(len(testset))

                ism_sum_recall += float(len(ism_hitset))/float(N)

        

            pm_avg_precision = pm_sum_precision/float(num_users_sample)

            pm_avg_recall = pm_sum_recall/float(num_users_sample)

    

            ism_avg_precision = ism_sum_precision/float(num_users_sample)

            ism_avg_recall = ism_sum_recall/float(num_users_sample)



            ism_avg_precision_list.append(ism_avg_precision)

            ism_avg_recall_list.append(ism_avg_recall)

    

            pm_avg_precision_list.append(pm_avg_precision)

            pm_avg_recall_list.append(pm_avg_recall)

            

        return (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list)

     



    #A wrapper method to calculate all the evaluation measures

    def calculate_measures(self, percentage):

        #Create a test sample of users

        self.create_user_test_sample(percentage)

        

        #Generate recommendations for the test sample users

        self.get_test_sample_recommendations()

        

        #Calculate precision and recall at different cutoff values

        #for popularity mode (pm) as well as item similarity model (ism)

        

        return self.calculate_precision_recall()

        #return (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list)    
import numpy as np

import pandas



#Class for Popularity based Recommender System model

class popularity_recommender_py():

    def __init__(self):

        self.train_data = None

        self.user_id = None

        self.item_id = None

        self.popularity_recommendations = None

        

    #Create the popularity based recommender system model

    def create(self, train_data, user_id, item_id):

        self.train_data = train_data

        self.user_id = user_id

        self.item_id = item_id



        #Get a count of user_ids for each unique song as recommendation score

        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()

        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)

    

        #Sort the songs based upon recommendation score

        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])

    

        #Generate a recommendation rank based upon score

        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        

        #Get the top 10 recommendations

        self.popularity_recommendations = train_data_sort.head(10)



    #Use the popularity based recommender system model to

    #make recommendations

    def recommend(self, user_id):    

        user_recommendations = self.popularity_recommendations

        

        #Add user_id column for which the recommendations are being generated

        user_recommendations['user_id'] = user_id

    

        #Bring user_id column to the front

        cols = user_recommendations.columns.tolist()

        cols = cols[-1:] + cols[:-1]

        user_recommendations = user_recommendations[cols]

        

        return user_recommendations

    



#Class for Item similarity based Recommender System model

class item_similarity_recommender_py():

    def __init__(self):

        self.train_data = None

        self.user_id = None

        self.item_id = None

        self.cooccurence_matrix = None

        self.songs_dict = None

        self.rev_songs_dict = None

        self.item_similarity_recommendations = None

        

    #Get unique items (songs) corresponding to a given user

    def get_user_items(self, user):

        user_data = self.train_data[self.train_data[self.user_id] == user]

        user_items = list(user_data[self.item_id].unique())

        

        return user_items

        

    #Get unique users for a given item (song)

    def get_item_users(self, item):

        item_data = self.train_data[self.train_data[self.item_id] == item]

        item_users = set(item_data[self.user_id].unique())

            

        return item_users

        

    #Get unique items (songs) in the training data

    def get_all_items_train_data(self):

        all_items = list(self.train_data[self.item_id].unique())

            

        return all_items

        

    #Construct cooccurence matrix

    def construct_cooccurence_matrix(self, user_songs, all_songs):

            

        ####################################

        #Get users for all songs in user_songs.

        ####################################

        user_songs_users = []        

        for i in range(0, len(user_songs)):

            user_songs_users.append(self.get_item_users(user_songs[i]))

            

        ###############################################

        #Initialize the item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

           

        #############################################################

        #Calculate similarity between user songs and all unique songs

        #in the training data

        #############################################################

        for i in range(0,len(all_songs)):

            #Calculate unique listeners (users) of song (item) i

            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]

            users_i = set(songs_i_data[self.user_id].unique())

            

            for j in range(0,len(user_songs)):       

                    

                #Get unique listeners (users) of song (item) j

                users_j = user_songs_users[j]

                    

                #Calculate intersection of listeners of songs i and j

                users_intersection = users_i.intersection(users_j)

                

                #Calculate cooccurence_matrix[i,j] as Jaccard Index

                if len(users_intersection) != 0:

                    #Calculate union of listeners of songs i and j

                    users_union = users_i.union(users_j)

                    

                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))

                else:

                    cooccurence_matrix[j,i] = 0

                    

        

        return cooccurence_matrix



    

    #Use the cooccurence matrix to make top recommendations

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):

        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        

        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.

        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])

        user_sim_scores = np.array(user_sim_scores)[0].tolist()

 

        #Sort the indices of user_sim_scores based upon their value

        #Also maintain the corresponding score

        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

    

        #Create a dataframe from the following

        columns = ['user_id', 'song', 'score', 'rank']

        #index = np.arange(1) # array of numbers for the number of samples

        df = pandas.DataFrame(columns=columns)

         

        #Fill the dataframe with top 10 item based recommendations

        rank = 1 

        for i in range(0,len(sort_index)):

            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:

                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]

                rank = rank+1

        

        #Handle the case where there are no recommendations

        if df.shape[0] == 0:

            print("The current user has no songs for training the item similarity based recommendation model.")

            return -1

        else:

            return df

 

    #Create the item similarity based recommender system model

    def create(self, train_data, user_id, item_id):

        self.train_data = train_data

        self.user_id = user_id

        self.item_id = item_id



    #Use the item similarity based recommender system model to

    #make recommendations

    def recommend(self, user):

        

        ########################################

        #A. Get all unique songs for this user

        ########################################

        user_songs = self.get_user_items(user)    

            

        print("No. of unique artists for the user: %d" % len(user_songs))

        

        ######################################################

        #B. Get all unique items (songs) in the training data

        ######################################################

        all_songs = self.get_all_items_train_data()

        

        print("no. of unique artists in the training set: %d" % len(all_songs))

         

        ###############################################

        #C. Construct item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        

        #######################################################

        #D. Use the cooccurence matrix to make recommendations

        #######################################################

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

                

        return df_recommendations

    

    #Get similar items to given items

    def get_similar_items(self, item_list):

        

        user_songs = item_list

        

        ######################################################

        #B. Get all unique items (songs) in the training data

        ######################################################

        all_songs = self.get_all_items_train_data()

        

        print("no. of unique artists in the training set: %d" % len(all_songs))

         

        ###############################################

        #C. Construct item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        

        #######################################################

        #D. Use the cooccurence matrix to make recommendations

        #######################################################

        user = ""

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

         

        return df_recommendations
train = pd.read_csv('../input/train.csv')



songs = pd.read_csv('../input//songs.csv')

test = pd.read_csv('../input//test.csv')

members = pd.read_csv('../input//members.csv')

songs_extra_info = pd.read_csv('../input/song_extra_info.csv')
df = train.merge(members,how='inner',on='msno')

df = df.merge(songs,how='inner',on='song_id')

df = df.merge(songs_extra_info,how='inner',on='song_id')

df.drop_duplicates(subset =['source_system_tab', 'source_screen_name',

       'source_type', 'target', 'city', 'bd', 'gender', 'registered_via',

       'registration_init_time', 'expiration_date', 'song_length', 'genre_ids',

       'artist_name', 'composer', 'lyricist', 'language'], 

                     keep = False, inplace = True) 

df.reset_index(drop = True,inplace=True)

df
sa=df[['msno','song_id','artist_name']]





sa = sa.groupby(['msno','artist_name']).count()

sa.reset_index(inplace =True)
sa.columns = ['user_id','song_id','listen_count']

sa
#df = sa.head(len(sa))

df = sa.head(50000)

song_grouped = df.groupby(['song_id']).agg({'listen_count': 'count'}).reset_index()

grouped_sum = song_grouped['listen_count'].sum()

song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100

song_grouped.sort_values(['listen_count', 'song_id'], ascending = [0,1])
###Fill in the code here

songs = sa['song_id'].unique()

len(songs)
train_data, test_data = train_test_split(sa, test_size = 0.95, random_state=0)

print(train_data.head(5))
pm = popularity_recommender_py()

pm.create(train_data, 'user_id', 'song_id')
id = members.iloc[6]['msno']

#user_id = users[5]

pm.recommend(id)
is_model = item_similarity_recommender_py()

is_model.create(train_data, 'user_id', 'song_id')
#Print the songs for the user in training data

user_id = members.iloc[7]['msno']

user_items = is_model.get_user_items(user_id)

#

print("------------------------------------------------------------------------------------")

print("Training data artists for the user msno: %s:" % user_id)

print("------------------------------------------------------------------------------------")



for user_item in user_items:

    print(user_item)



print("----------------------------------------------------------------------")

print("Recommendation process going on:")

print("----------------------------------------------------------------------")

#7

#Recommend songs for the user using personalized model

is_model.recommend(user_id)
#Print the songs for the user in training data

user_id = members.iloc[7]['msno']

user_items = is_model.get_user_items(user_id)

#

print("------------------------------------------------------------------------------------")

print("Training data artists for the user msno: %s:" % user_id)

print("------------------------------------------------------------------------------------")



for user_item in user_items:

    print(user_item)



print("----------------------------------------------------------------------")

print("Recommendation process going on:")

print("----------------------------------------------------------------------")



is_model.recommend(user_id)