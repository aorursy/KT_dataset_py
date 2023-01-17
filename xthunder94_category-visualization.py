%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import itertools

import collections

import operator



data = pd.read_csv("../input/anime.csv", encoding="ISO-8859-1")
# Normalize genres

data = data.replace({'Harem': 'Hentai'}, regex=True)

data = data.replace({'Ecchi': 'Hentai'}, regex=True)

data = data.replace({'Shoujo Ai': 'Hentai'}, regex=True)

data = data.replace({'Yaoi': 'Hentai'}, regex=True)

data = data.replace({'Yuri': 'Hentai'}, regex=True)

data = data.replace({'Shounen Ai': 'Hentai'}, regex=True)



data = data.replace({'Demons': 'Vampire'}, regex=True)



data = data.replace({'Supernatural': 'Magic'}, regex=True)

data = data.replace({'Super Power': 'Magic'}, regex=True)

data = data.replace({'Sci-Fi': 'Magic'}, regex=True)
# Find all genres represented

genres = set()

for entry in data['genre']:

    if not type(entry) is str:

        continue

    genres.update(entry.split(", "))

print(genres)

print("Total Genres: " + str(len(genres)))
# List genres by count

genres_count = collections.defaultdict(int)

for entry in data['genre']:

    if not type(entry) is str:

        continue

    seen_already = set()

    for genre in entry.split(", "):

        if genre in seen_already:

            continue

        seen_already.add(genre)

        genres_count[genre] += 1

sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)
# Plot all animes by rating and popularity colored by genre

fig = plt.figure(figsize=(20,20))

ax = plt.gca()

plt.title('All Animes Rating vs. Popularity By Genre')

plt.xlabel('Rating')

plt.ylabel('Popularity (People)')

num_colors = len(genres)

cm = plt.get_cmap('gist_rainbow')

ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

ax.set_yscale('log')

# For each genre, plot data point if it falls in that category

for genre in genres:

    data_genre = data[data.genre.str.contains(genre) == True]

    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12, label=genre)

ax.legend(numpoints=1, loc='upper left');
# Plot select genres by rating and popularity colored by genre

genre_plot = ['Shounen', 'Shoujo', 'Seinen', 'Josei', 'Kids']



fig = plt.figure(figsize=(10,10))

ax = plt.gca()

plt.title('All Animes Rating vs. Popularity By Genre')

plt.xlabel('Rating')

plt.ylabel('Popularity (People)')

num_colors = len(genre_plot)

cm = plt.get_cmap('gist_rainbow')

ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

ax.set_yscale('log')

# For each genre, plot data point if it falls in that category

for genre in genre_plot:

    data_genre = data[data.genre.str.contains(genre) == True]

    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12, label=genre)

ax.legend(numpoints=1, loc='upper left');
# One vs. each other genre pairwise graphs by rating and popularity colored by genre

g1 = "Comedy"



for g2 in genres:

    if g1 == g2:

        continue

    fig = plt.figure(figsize=(10,10))

    ax = plt.gca()

    plt.title(g1 + " (Red) vs. " + g2 + " (Green)")

    plt.xlabel('Rating')

    plt.ylabel('Popularity (People)')

    num_colors = 2

    cm = plt.get_cmap('gist_rainbow')

    ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

    ax.set_yscale('log')

    # For each genre, plot data point if it falls in that category

    data_genre = data[data.genre.str.contains(g1) == True]

    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12)

    data_genre = data[data.genre.str.contains(g2) == True]

    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12)

    plt.show()