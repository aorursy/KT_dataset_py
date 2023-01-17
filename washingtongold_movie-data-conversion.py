import pandas as pd

import progressbar as pb

import random

data = pd.read_csv('/kaggle/input/movie-reviews/train.csv')

data.head()
c_data = pd.DataFrame()

movies = []

c_data['id'] = 0

for movie in pb.progressbar(data['movie']):

    if movie not in movies:

        movies.append(movie)

        c_data[movie] = 0

users = []

for user in pb.progressbar(data['user']):

    if user not in users:

        users.append(user)

        append_dic = {'id':user} 

        for column in c_data.columns:

            if column != 'id':

                append_dic[column] = 0

        c_data = c_data.append(pd.DataFrame([append_dic]))

c_data = c_data.set_index('id')
for index in pb.progressbar(range(len(data)-1)):

    c_data.loc[data['user'][index],data['movie'][index]] = data['rating'][index]
c_data
c_data.to_csv ('converted.csv', index = None, header=True)