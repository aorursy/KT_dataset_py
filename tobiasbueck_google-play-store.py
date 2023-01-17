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
df = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
df.head()
ratings = list(df["Rating"].dropna())
ratings[:10]
from matplotlib import pyplot as plt
ratings_float = list(map(lambda x: float(x), ratings))
ratings_float[:10]
fig1, ax1 = plt.subplots()

ax1.set_title('Box Plot Ratings')

ax1.boxplot(ratings_float, vert=False)
filtered_ratings = list(filter(lambda x: x <= 5, ratings))
histogramm_ratings = list([i / 10 for i in range(11, 51)])

# rgb ()

colors = list((1 / ((0.01 * i) + 1), 1 - 1 / ((0.1 * i) + 1), 0.2) for i in range(len(histogramm_ratings)))
fig1, ax1 = plt.subplots()

n, bins, patches,  = plt.hist(filtered_ratings, bins=histogramm_ratings, rwidth=1)

ax1.set_xlabel("Rating")

ax1.set_ylabel("Amount")

ax1.set_title("Ratings Verteilung")

for bar, color in zip(patches, colors):

    bar.set_color(color)
categories = list(df["Category"])
unique_categories = list(set(categories))
print(unique_categories)
categories_count = {}

for c in unique_categories:

    categories_count[c] = categories.count(c)
categories_count
total = sum(list(categories_count.values()))
for c in categories_count.keys():

    categories_count[c]= categories_count[c] / total
categories_count
del categories_count["1.9"]
fig, ax = plt.subplots(figsize=(15, 10))

ax.pie(list(categories_count.values()), labels=list(categories_count.keys()), rotatelabels = True)
## Categories with the most interception
genres = list(df["Genres"])
genres[:5]
all_genres = []
for g in genres:

    all_genres.extend(g.split(";"))
list(set(all_genres))
genres_cut = {g:{gen: 0 for gen in all_genres} for g in all_genres}
print(genres_cut)
for g in genres:

    all_gs = g.split(";")

    for gen in all_gs:

        for gen2 in all_gs:

            genres_cut[gen][gen2] += 1

    
abs_genres_cutt = genres_cut.copy()
for gen in genres_cut.keys():

    m = max(genres_cut[gen].values())

    for k_gen2 in genres_cut[gen].keys():

        genres_cut[gen][k_gen2] /= m

genres_cut["Weather"]
## How Art& Design similar are the genres
similars = []



for gen_name, gen_similars in genres_cut.items():

    similars.append((gen_name, gen_similars["Art & Design"]))
similars_sort = reversed(sorted(similars ,key=lambda x: x[1]))
for s in similars_sort:

    print(s)
def get_similarity(gen1, gen2):

    return genres_cut[gen1][gen2]
def create_groups(sensitivity):

    groups = []

    for gen in list(genres_cut.keys()):

        for group in groups:

            # average_similarity = sum(map(lambda x: get_similarity(gen, x), group)) / len(group)

            max_similarity = max(map(lambda x: get_similarity(gen, x), group))

            if max_similarity > sensitivity:

                group.append(gen)

                break

        else:

            groups.append([gen])

    return groups
groups = create_groups(0.1)

print(groups)
circle1 = plt.Circle((1, 3), 2)

circle2 = plt.Circle((4, 6), 2)

circle3 = plt.Circle((8, 5), 2)





fig, ax = plt.subplots()

ax.set_xticks(list([i for i in range(-2, 12)]))

ax.set_yticks(list([i for i in range(12)]))



ax.text(-0.5,3, "Art & Design")

ax.text(0,2, "Creativity")





ax.text(3.2,6, "Pretend Play")

ax.text(3,5, "Casual")

ax.text(3,7, "Educational")



ax.text(6.5,5, "Action&Adventure")

ax.text(7.5,4, "Adventure")

ax.text(7.8,6, "Racing")

ax.add_artist(circle1)

ax.add_artist(circle2)

ax.add_artist(circle3)

ax.set_title("Similar App Genres")