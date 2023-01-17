import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")
posts = pd.read_csv('../input/post.csv', parse_dates=['timeStamp'])

comments = pd.read_csv('../input/comment.csv', parse_dates=['timeStamp'])

# members = pd.read_csv('../input/member.csv')

# likes = pd.read_csv('../input/like.csv')
posts['hour'] = pd.DatetimeIndex(posts['timeStamp']).hour

posts["post_count"] = 1
allowedColumns = ['hour', 'shares', 'likes', 'post_count']

posts_filtered = posts[allowedColumns]

posts_by_hour = posts_filtered.groupby(["hour"]).sum().reset_index()

posts_by_hour.head(2)
f, ax = plt.subplots(figsize=(12, 7))

plt.xticks(list(range(0,24)))

plt.yticks([0, 300, 700, 1000, 2000, 4000, 5000])

posts_by_hour.plot("hour", "post_count", ax=ax)

posts_by_hour.plot("hour", "likes", ax=ax)

posts_by_hour.plot("hour", "shares", ax=ax)
f, ax = plt.subplots(figsize=(12, 7))

posts_by_hour[["post_count","likes", "shares"]].plot.barh(stacked=True, ax=ax)
comments['hour'] = pd.DatetimeIndex(comments['timeStamp']).hour

comments['comment_count'] = 1
allowedColumns = ['hour', 'comment_count']

comments_filtered = comments[allowedColumns]

comments_by_hour = comments_filtered.groupby(["hour"]).sum().reset_index()

comments_by_hour.head(2)
f, ax = plt.subplots(figsize=(12, 7))

comments_by_hour.plot("hour", "comment_count", ax=ax, color='b', legend=False)

comments_by_hour.plot("hour", "comment_count", ax=ax, kind='bar')