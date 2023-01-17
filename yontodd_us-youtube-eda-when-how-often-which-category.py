import kaggle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
%matplotlib inline
sns.set_style("darkgrid")
sns.set_context("paper",font_scale=1)
plt.rcParams['figure.figsize'] = (10, 10)
data = pd.read_csv("../input/USvideos.csv")
data.head()
# Infomation about the data
data.info()
# Unique objects?

data["video_id"].nunique()
type(data["trending_date"][0])
data["trending_date"] = pd.to_datetime(data["trending_date"],format="%y.%d.%m")
data["trending_date"].head()
type(data["publish_time"][0])
data["publish_time"] = pd.to_datetime(data["publish_time"],format="%Y-%m-%dT%H:%M:%S.%fZ")
data["publish_time"].head()
# Separate date from time for publish

data["publish_date"] = pd.DatetimeIndex(data["publish_time"]).date
data["publish_date"] = pd.to_datetime(data["publish_date"],format="%Y-%m-%d")
data["publish_date"].head()
data["publish_time_of_day"] = pd.DatetimeIndex(data["publish_time"]).time
data["publish_time_of_day"].head()
# Hour published
data["publish_hour"] = data["publish_time_of_day"].apply(lambda x: x.hour)
data["publish_hour"].head()
# Add column of days between published and trended

# How about just videos that trended the day of/after they were published

# Pull days trended for each
# Remember, can be zero, if it trends on the day it was published
data["days_trended_after_publish"] = data["trending_date"] - data["publish_date"]
# Change days_trended_after_publish to an integer
data["days_trended_after_publish"] = data["days_trended_after_publish"].dt.days
# Add column with sum of total days trended per video
trended_count = data.groupby("video_id").count()["days_trended_after_publish"].reset_index()
trended_count.columns = ["video_id","trended_count"]
data = data.merge(trended_count,on="video_id")
# Pull category ID data

id_to_category = {}

with open("../input/US_category_id.json","r") as f:
    id_data = json.load(f)
    for category in id_data["items"]:
        id_to_category[category["id"]] = category["snippet"]["title"]

id_to_category
type(data["category_id"][0])
data["category_id"] = data["category_id"].astype(str)
# Map that data onto the dataset
data.insert(4, "category",data["category_id"].map(id_to_category))
# Ok, let's check out the data with the changes we made
data[data["video_id"] == "2kyS6SvSYSE"]
sns.heatmap(data.corr(),cmap="Blues",annot=True)
# Note, this might not be a good proxy of much, since some videos are in the dataset multiple times
# Correlation matrix for only videos on the last day trended
sns.heatmap(data[data["trended_count"] == data["days_trended_after_publish"]].corr(),cmap="Blues",annot=True)
# Sum of views per video by hour published
sns.barplot(data = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").sum()["views"].reset_index(),x="publish_hour",y="views",palette="coolwarm")
# Total days trended by hour published
sns.barplot(x="publish_hour",y="trended_count",data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").sum()["trended_count"].reset_index(),palette="coolwarm")
sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").mean()["trended_count"].reset_index(),
            x="publish_hour",y="trended_count",palette="coolwarm")
sns.boxplot(data=data[data["trended_count"] == data["days_trended_after_publish"]],
            x="publish_hour",y="trended_count",palette="coolwarm")
# Add column for day of week that each video was published
data["publish_day"] = data["publish_date"].apply(lambda x: x.strftime('%A'))
sns.barplot(data=data.groupby("publish_day").sum()["trended_count"].sort_values(ascending=False).reset_index(),x="publish_day",y="trended_count",palette="coolwarm")
sns.barplot(data=data.groupby("publish_day").mean()["trended_count"].sort_values(ascending=False).reset_index(),x="publish_day",y="trended_count",palette="coolwarm")
# Unique videos
unique_videos = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").nunique()["video_id"].reset_index()
unique_videos.sort_values(by="video_id",ascending=False).head()
# Total days between first and last published date
last_published = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").max()["publish_date"].reset_index()
last_published.head()
first_published = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").min()["publish_date"].reset_index()
first_published.head()
consistency = first_published.merge(last_published,on="channel_title")
consistency.columns = ["channel_title","first_published","last_published"]
consistency["total_days"] = consistency["last_published"] - consistency["first_published"]
consistency["total_days"] = consistency["total_days"].dt.days
consistency = consistency.merge(unique_videos,on="channel_title")
consistency["average_days_between_videos"] = consistency["total_days"]/consistency["video_id"]
consistency = consistency.merge(data[["video_id","channel_title","trended_count"]].drop_duplicates().groupby(by="channel_title").sum()["trended_count"].reset_index(),on="channel_title")
consistency.head()
sns.lmplot(data=consistency,x="average_days_between_videos",y="trended_count")
x=5
sns.lmplot(data=consistency[consistency["video_id"] >=x],x="average_days_between_videos",y="trended_count")
consistency["average_days_trended"] = consistency["trended_count"]/consistency["video_id"]
consistency.head()
sns.lmplot(data=consistency,x="average_days_between_videos",y="average_days_trended")
sns.lmplot(data=consistency[consistency["video_id"] >=2],x="average_days_between_videos",y="average_days_trended")
data["likes-to-dislikes"] = data["likes"]/data["dislikes"]
sns.heatmap(data[["views","likes","dislikes","comment_count","likes-to-dislikes","trended_count"]].corr(),cmap="Blues",annot=True)
# Note: correlations taking into account only the last day a video trended doesn't do much to correlations
sns.heatmap(data[data["days_trended_after_publish"] == data["trended_count"]].corr(),cmap="Blues",annot=True)
sns.barplot(data=data.groupby("category").count()["views"].reset_index().sort_values(by="views",ascending=False),x="views",y="category",palette="coolwarm")
sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby(by="category").sum()["trended_count"].reset_index().sort_values(by="trended_count",ascending=False),y="category",x="trended_count",palette="coolwarm")
sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby(by="category").mean()["trended_count"].reset_index().sort_values(by="trended_count",ascending=True),x="trended_count",y="category",palette="coolwarm")