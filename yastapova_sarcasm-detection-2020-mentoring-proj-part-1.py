# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualizations and charts



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/sarcasm/train-balanced-sarcasm.csv")

data.head(10)
data.isna().sum(axis=0)
empty_comments = data["comment"].isna()

empty_comments = data[empty_comments].index

data.drop(empty_comments, axis=0, inplace=True)
for c in data[data["label"] == 1]["comment"][0:10]:

    print(c)
data.describe()
data["downs"].value_counts()
data["ups"].value_counts()
sum(data["ups"] < 0)
i = data["ups"].idxmin()

data.loc[i, :]
data["label"].value_counts()
data.head(15)
len(data["subreddit"].unique())
top_subreddits = data["subreddit"].value_counts()[0:25]

top_subreddits = list(top_subreddits.index)
data[data["subreddit"].isin(top_subreddits)].groupby("subreddit").agg({"label" : "mean"})
data.groupby("label").agg({"score" : "mean"})
scores_sarc = data["score"][data["label"] == 1]

scores_non = data["score"][data["label"] == 0]

bins = list(range(-15, 16))

plt.hist(scores_sarc, bins=bins, alpha=0.5, label="sarcastic")

plt.hist(scores_non, bins=bins, alpha=0.5, label="non-sarcastic")

plt.xlabel("score")

plt.ylabel("frequency")

plt.legend(loc="upper right")

plt.show()
by_month = data.groupby("date").agg({"label" : "mean", "comment" : "count"})

by_month
months = list(by_month.index)

label_pos = list(range(0, len(months), 6))

m_labels = [months[i] for i in label_pos]



plt.plot(months, by_month["label"])

plt.xlabel("year-month")

plt.ylabel("% sarcastic")

plt.xticks(label_pos, m_labels, rotation=45, ha="right")

plt.show()
months = list(by_month.index)

label_pos = list(range(0, len(months), 6))

m_labels = [months[i] for i in label_pos]



plt.plot(months, by_month["comment"])

plt.xlabel("year-month")

plt.ylabel("# of comments")

plt.xticks(label_pos, m_labels, rotation=45, ha="right")

plt.show()
data["comment_length"] = data["comment"].apply(lambda x: len(x))

data.head()
data.groupby("label").agg({"comment_length" : "mean"})
data.groupby("label").agg({"comment_length" : "std"})
data["date_year"] = data["date"].str[0:4]

data["date_month"] = data["date"].str[5:]

data["date_year"] = data["date_year"].astype("int64")

data["date_month"] = data["date_month"].astype("int64")

data.head()
data.groupby("date_month").agg({"label" : "mean", "comment" : "count"})
import datetime

date_format = "%Y-%m-%d %H:%M:%S"



def get_weekday(d):

    d = datetime.datetime.strptime(d, date_format)

    return d.strftime("%w")



data["comment_day"] = data["created_utc"].apply(lambda x: get_weekday(x))

data["comment_day"] = data["comment_day"].astype("int64")

data.head()
data.groupby("comment_day").agg({"label" : "mean", "comment" : "count"})
data.to_csv("sarcasm_prepped_data.csv", index=False)