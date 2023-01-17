import pandas as pd

import itertools as it

import collections as co

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
anime = pd.read_csv("../input/anime-recommendations-database/anime.csv", header=0, index_col=0, encoding='utf-8')
anime['genre'] = anime['genre'].fillna('None')

anime['genre'] = anime['genre'].apply(lambda x: x.split(', '))  # split in according to ‘， ’
genre = it.chain(*anime['genre'])  # spread list

genre_count = co.Counter(genre)



genreframe = pd.DataFrame.from_dict(genre_count, orient='index').reset_index().rename(columns={'index':'genre', 0: 'count'})

genreframe = genreframe.sort_values('count', ascending=False)
plt.figure(figsize=(10, 12))

plt.barh(genreframe["genre"].head(20), genreframe["count"].head(20),

         facecolor='tan', height=0.5, edgecolor='r', alpha=0.6)

plt.yticks(fontsize=10)

plt.xticks(fontsize=10)

plt.title(label=" Genre_Ranking Top 20", loc="center")





plt.show()
type = pd.DataFrame(anime["type"])

type = type.groupby("type").aggregate({"type": "count"}).rename(columns={'type': 'count', 'index': 'type'})

type = type.sort_values('count', ascending=False).reset_index()



plt.figure(figsize=(10, 10))

explode = [0.07,0,0,0,0,0]

labels = type["type"]

colors = ['aqua', 'yellow', 'limegreen', 'orange', 'cornflowerblue', 'slategray']

plt.pie(type["count"], autopct='%.1f%%', explode=explode, labels=labels, colors=colors)

plt.title(label=" Type of Animes", loc="center", fontsize=15)



plt.show()
type = pd.DataFrame(anime[["type", "rating"]]).dropna()

type = type.sort_values('type', ascending=False).reset_index()
plt.figure(figsize=(8, 10))

sns.boxplot(x='type', y='rating', data=type)

plt.yticks(fontsize=12)

plt.xticks(fontsize=12)

plt.title(label=" Types VS Rating", loc="center", fontsize=18)



plt.show()
type = pd.DataFrame(anime[["name", "type", "rating"]]).dropna()

type = type.sort_values(['type', 'rating'], ascending=False)

group = type.groupby(type["type"]).aggregate({"rating": "max", "name": "count"})
max_TV = type.iloc[0:6]

max_special = type.iloc[3671:3676]

max_OVA = type.iloc[5342:5347]

max_ONA = type.iloc[8627:8632]

max_music = type.iloc[9279:9284]

max_movie = type.iloc[9767:9772]

data = pd.concat([max_TV, max_movie, max_OVA, max_ONA, max_special, max_music], ignore_index=True)
print(data)
TV_anime = anime[anime['type'] == 'TV']



for index, lst in zip(TV_anime.index,TV_anime['genre'].values):

    for genre in lst:

        TV_anime.at[index, genre] = 1



TV_anime = TV_anime.fillna(0).drop(columns="None")
TV_Action = TV_anime[TV_anime['Action'] == 1].sort_values(['rating'], ascending=False)

TV_Action = TV_Action.head(1)

TV_Action = pd.DataFrame(TV_Action[["name", "rating"]])



TV_Adventure = TV_anime[TV_anime['Adventure'] == 1].sort_values(['rating'], ascending=False)

TV_Adventure = TV_Adventure.head(1)

TV_Adventure = pd.DataFrame(TV_Adventure[["name", "rating"]])



TV_Comedy = TV_anime[TV_anime['Comedy'] == 1].sort_values(['rating'], ascending=False)

TV_Comedy = TV_Comedy.head(1)

TV_Comedy = pd.DataFrame(TV_Comedy[["name", "rating"]])



TV_SciFi = TV_anime[TV_anime['Sci-Fi'] == 1].sort_values(['rating'], ascending=False)

TV_SciFi = TV_SciFi.head(1)

TV_SciFi = pd.DataFrame(TV_SciFi[["name", "rating"]])



TV_School = TV_anime[TV_anime['School'] == 1].sort_values(['rating'], ascending=False)

TV_School = TV_School.head(1)

TV_School = pd.DataFrame(TV_School[["name", "rating"]])



TV_Thriller = TV_anime[TV_anime['Thriller'] == 1].sort_values(['rating'], ascending=False)

TV_Thriller = TV_Thriller.head(1)

TV_Thriller = pd.DataFrame(TV_Thriller[["name", "rating"]])

data = pd.concat([TV_Action, TV_Adventure, TV_Comedy, TV_School, TV_SciFi, TV_Thriller], ignore_index=True)

data["genre"] = ["TV_Action", "TV_Adventure", "TV_Comedy", "TV_School", "TV_SciFi", "TV_Thriller"]
print(data)