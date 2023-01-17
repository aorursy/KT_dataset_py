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
df = pd.read_csv("/kaggle/input/youtube-new/DEvideos.csv")
df.head()
df.describe()
views = list(df["views"].dropna())
from matplotlib import pyplot as plt
sorted_views = list(reversed(sorted(views)))
sorted_views[:10]
bin_count = []
fig1, ax1 = plt.subplots()

ax1.boxplot(views)
bin_edges = []

for i in range(30):

    edge = 1.4** i + 200_000 * i 

    bin_edges.append(edge)
plt.hist(views, bins=bin_edges)
fig1, ax1 = plt.subplots()

ax1.set_title("Youtube Video Views")

ax1.set_xlabel("Views")

ax1.set_ylabel("Count")

ax1.hist(views, bins=bin_edges)
# which tags or description, kind of videos get in trends
tags = list(df["tags"].dropna())
all_tags = []
for tag in tags:

    tags_splitted = tag.split("|")

    all_tags.extend(tags_splitted)
all_tags = list(map(lambda x: x.replace("\"", ""), all_tags))
all_tags[:10]
unique_tags = list(set(all_tags))
count_tag = {unique_tag:0 for unique_tag in unique_tags }
for tag in all_tags:

    count_tag[tag] += 1
del count_tag['[none]']
counts = list(count_tag.items())
counts.sort(key=lambda x: x[1], reverse=True)
counts[:10]
fig1, ax1 = plt.subplots()

xs = list(range(10))

most_used_tags = counts[:10]

ys = list(reversed(list(map(lambda x: x[1], most_used_tags))))

labels = list(reversed(list(map(lambda x: x[0], most_used_tags))))

ax1.set_title("Most trending youtube tags in germany")

ax1.set_xlabel("times trended")

ax1.barh(xs, ys, tick_label=labels)
### Youtubers
df.head()
channels = list(df["channel_title"])
unique_channels = list(set(channels))
channels_count = {c: 0 for c in unique_channels}
for c in channels:

    channels_count[c] += 1

    
sorted_channels = list(reversed(sorted(list(channels_count.items()), key = lambda x: x[1])))
sorted_channels[:10]
channels_most_trending = sorted_channels[:10]
fig1, ax1 = plt.subplots()

xs = list(range(10))

ys = list(reversed(list(map(lambda x: x[1], channels_most_trending))))

labels = list(reversed(list(map(lambda x: x[0], channels_most_trending))))

ax1.set_title("Most trending youtube channels in Germany")

ax1.set_xlabel("times trended")

ax1.barh(xs, ys, tick_label=labels)
df.head()
views_likes = df[["views", "likes"]]
views_likes.head()
views_likes = views_likes.dropna()
views_likes.sort()
views = list(views_likes["views"])

likes = list(views_likes["likes"])
v_l = list(sorted(list(zip(views, likes)), key= lambda x: x[0]))
sort_views, sort_likes = zip(*v_l)
fig1, ax1 = plt.subplots(figsize=(10,8))

ax1.set_title("Correlation Views/Likes")

ax1.set_xlabel("Views")

ax1.set_ylabel("Likes")

ax1.scatter(sort_views, sort_likes, s=0.4)
import numpy as np

from sklearn.linear_model import LinearRegression
x = np.array(sort_views).reshape((-1, 1))

y = np.array(sort_likes)
model = LinearRegression()

model.fit(x,y)
print(model.intercept_)

print(model.coef_)
# Im Durschnitt hat ein Video 3% likes, der views
test_x = list(range(0, 100_000_000, 10_000))
test_y = model.predict(np.array(test_x).reshape((-1, 1)))
fig1, ax1 = plt.subplots(figsize=(10,8))

ax1.set_title("Correlation Views/Likes")

ax1.set_xlabel("Views")

ax1.set_ylabel("Likes")

ax1.plot(test_x, test_y)

ax1.scatter(sort_views, sort_likes, s=0.4)
vid_popularity = df[["views","likes", "dislikes"]].dropna()
views = list(vid_popularity["views"])

likes = list(vid_popularity["likes"])

dislikes = list(vid_popularity["dislikes"])
like_dislike = list(map(lambda x: x[0] / (x[1] + 1),zip(likes, dislikes)))
like_dislike
df["like/dislike"] = like_dislike
df.head()
df["like/dislike"].describe()
df[df["like/dislike"] < 200]["like/dislike"].hist(bins=200)
views_likes = list(map(lambda x: (x[0] / (x[1] + 1)) * 100,zip(likes, views)))
df["likes%/views"] = views_likes
df = df.drop("views/likes", axis=1)
df.head()
df["likes%/views"].hist(bins = 100)