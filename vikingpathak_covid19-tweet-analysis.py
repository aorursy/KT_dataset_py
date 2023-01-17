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
dataframe = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
dataframe.head()
dataframe.isnull().sum()
verified_tweets = dataframe.groupby(['user_verified'])[["user_name"]].count().reset_index()
print(verified_tweets.head())

unverified_users = verified_tweets.at[0,"user_name"]   #count of unverified users
verified_users = verified_tweets.at[1,"user_name"]     #count of verified users
import matplotlib.pyplot as plt
labels = ["Verified Users", "Un-verified Users"]
user_count = [verified_users, unverified_users]

fig= plt.figure(figsize=(6,6))

plt.pie(user_count,labels=labels,autopct='%1.1f%%', explode=(0.1, 0))
plt.title('Tweet from verified users')
location_data = dataframe.groupby(['user_location'])[['user_name']].count()
top_10_locations = location_data.sort_values(by = 'user_name', ascending=False).head(10).reset_index()
top_10_locations
fig = plt.figure(figsize=(10,6))
plt.barh(data = top_10_locations.iloc[::-1], y = 'user_location', width = 'user_name')
plt.title("Top 10 locations with most tweets")
source_count = dataframe.groupby('source')[['user_name']].count()
source_count = source_count.sort_values(by="user_name").tail(10).reset_index()
source_count
fig = plt.figure(figsize=(10,6))
plt.barh(data=source_count, y="source", width="user_name")
plt.title("Top 10 sources of highest tweet")
hashtags = dataframe[['user_name', 'hashtags']].dropna()
hashtag_count = {}
for hashtag_str in hashtags['hashtags']:
    hashtag_list = eval(hashtag_str)
    for hashtag in hashtag_list:
        if hashtag not in hashtag_count.keys():
            hashtag_count[hashtag] = 1
        else:
            hashtag_count[hashtag] += 1
top_10_hashtags = sorted(hashtag_count.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:10]
top_10_hash_df = pd.DataFrame(data = top_10_hashtags, columns = ['hashtag', 'count'])
top_10_hash_df