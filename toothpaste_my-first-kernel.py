import numpy as np 

import pandas as pd

from random import randint





import matplotlib.pyplot as plt

import seaborn as sns

import colorcet as cc
anime = pd.read_csv('../input/anime.csv')

anime.loc[anime.episodes.isin(['Unknown']), 'episodes'] = -1

anime.episodes = anime.episodes.astype('int32')

anime.genre.fillna('')



users = pd.read_csv('../input/rating.csv')

users = pd.merge(users, anime, on = 'anime_id')
plt.rcParams['figure.figsize'] = (8, 6)



hm = sns.heatmap(anime.corr(), annot = True, linewidth = .5, cmap = 'Blues')

hm.set_title(label = 'Heatmap')



hm
fig, ax = plt.subplots()



ax.scatter(anime.members, anime.rating, s = 10, alpha = .5, c = anime.members, cmap = cc.cm['bkr'])

ax.set_title('Rating x Members')

ax.set_xlabel('Members')

ax.set_ylabel('Rating')

ax.grid(True, alpha = .5)



fig.show()
episodes_count = anime.sort_values('episodes', ascending = False)

episodes_count = episodes_count.reset_index().drop('anime_id', axis = 1)

episodes_count = episodes_count[episodes_count.episodes >= 0]

episodes_count = episodes_count[episodes_count.type == 'TV']

episodes_count = episodes_count[episodes_count.episodes <= 100]

episodes_count = episodes_count.groupby(['episodes'])['episodes'].count()



fig, ax = plt.subplots()



ax.plot(episodes_count.index, episodes_count, linewidth = 1, c = 'k')



ax.set_title('Animes x Episodes')

ax.set_xlabel('Episodes')

ax.set_xticks([0, 12, 20, 26, 30, 40, 52, 60, 70, 80, 90, 100])

ax.set_ylabel('Animes')

ax.grid(True, alpha = .5)



fig.show()
ranking_members = anime.sort_values('members', ascending = False).iloc[0:20]

ranking_members = ranking_members.sort_values('members', ascending = False)

ranking_members = ranking_members.reset_index().drop('index', axis = 1)



ranking_members.index = ranking_members.index + 1



print('\nRanking de Popularidade')

print(ranking_members.loc[:, ['name', 'members', 'rating']])
def user_info(x):

    user = users.loc[users.user_id == x]

    

    user_anime = user.sort_values('rating_x', ascending = False)

    user_anime = user_anime.loc[user_anime.rating_x >= 0]

    user_anime = user_anime.iloc[0:50]

    

    user_types = pd.Series(user.groupby('type')['type'].count())

    user_types = user_types.sort_values(ascending = False)

    

    fig, ax = plt.subplots(figsize = (10, 10))

    

    ax.pie(user_types, labels = user_types.index, autopct = '%1.1f%%')

    ax.set_title('type user ' + str(x), fontsize = 22)

    

    fig.show() 

 

    def genre_split(x):

        return x.split(', ')

    

    user_genre = []

    genre = user.loc[:, 'genre']

    genre = genre.apply(genre_split)

    

    for gen in genre:

        for index, g in enumerate(gen):

            user_genre.append(g)

            

    user_genre = pd.DataFrame(data = user_genre, columns = ['genre'])

    user_genre = pd.Series(user_genre.groupby('genre')['genre'].count())

    user_genre = user_genre.sort_values(ascending = False)

    

    fig, ax = plt.subplots(figsize = (10, 10))

    

    ax.pie(user_genre, labels = user_genre.index)

    ax.set_title('genre user ' + str(x), fontsize = 22)

    

    fig.show()    

    

    return user_anime, user_types, user_genre

    

u_anime, u_types, u_genre = user_info(randint(1, users.user_id.max()))
print('User\'s top animes')

print('{}\n'.format(u_anime.iloc[0:20, [3, 2]]))



print('User types')

print('{}\n'.format(u_types))



print('User genre')

print('{}\n'.format(u_genre))