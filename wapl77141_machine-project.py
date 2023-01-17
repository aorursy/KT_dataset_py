import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


ca = pd.read_csv('../input/CAvideos.csv', index_col=0, parse_dates=True, skipinitialspace=True)
ca['trending_date'] = pd.to_datetime(ca['trending_date'], format='%y.%d.%m')

publish_time = pd.to_datetime(ca['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
ca['publish_date'] = publish_time.dt.date
ca['publish_time'] = publish_time.dt.time
ca['publish_hour'] = publish_time.dt.hour


mean=np.mean(ca.views)
median=np.median(ca.views)
std=np.std(ca.views)

print("mean: ")
print(mean)
print("median: ")
print(median)
print("Standard Deviation")
print(dev)
ca.head(1)
ca.columns
id_dict = {}

with open('../input/CA_category_id.json', 'r') as file:
    data = json.load(file)
    for category in data['items']:
        id_dict[int(category['id'])] = category['snippet']['title']

ca['category_id'] = ca['category_id'].map(id_dict)
ca.head(3)
plt.figure(figsize=(12,6))
sns.countplot(x='category_id',data=ca,palette='Set1', order=ca['category_id'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Video Categories')
plt.ylabel('Total Videos by Categories')
plt.title('Total trending videos by Category in Canada')

plt.tight_layout()
plt.show()
views_by_category_total = ca.groupby('category_id')['views'].sum().sort_values(ascending=False)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.barplot(y=views_by_category_total.index, x=(views_by_category_total/1000000), palette='Set1')
plt.xlabel('Total Views by Categories (millions)')
plt.ylabel('Video Categories')
plt.title('Total Views of Trending videos by Category in Canada\n(in millions)')
views_by_category=ca.groupby('category_id')['views'].std()
print(views_by_category)
views_by_category_avg = ca.groupby('category_id')['views'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,8))
plt.subplot(1,2,2)
sns.barplot(y=views_by_category_avg.index, x=(views_by_category_avg/1000000), palette='Set1')
plt.xlabel('AVG Views by Categories (millions)')
plt.ylabel('')
plt.title('Average Views of trending videos by Category in Canada\n(in millions)')


most_viewed_entertainment = ca[ca['category_id']=='Entertainment'][['title','category_id','views']].sort_values(by='views', ascending=False)
most_viewed_entertainment = most_viewed_entertainment.groupby('title')['views'].mean().sort_values(ascending=False).head(1)
most_viewed_entertainment


least_viewed_entertainment = ca[ca['category_id']=='Entertainment'][['title','category_id','views']].sort_values(by='views', ascending=True)
least_viewed_entertainment = least_viewed_entertainment.groupby('title')['views'].mean().sort_values(ascending=True).head(1)
least_viewed_entertainment

plt.figure(figsize=(12,6))
sns.regplot(ca['views']/1000000, ca['likes']/1000, color='#55A868')

plt.title('Scatter Plot and Regression for Likes over Views')
plt.xlabel('Views (in millions)')
plt.ylabel('Likes (in thousands)')
plt.legend(['Likes'])

plt.show()


cor=np.corrcoef(ca.views,ca.likes)
print("correlation of likes and views: ")
print(cor)
plt.figure(figsize=(12,6))
sns.regplot(ca['views']/1000000, ca['dislikes']/1000, color='#B66468')

plt.title('Scatter Plot and Regression for Dislikes over Views')
plt.xlabel('Views (in millions)')
plt.ylabel('Dislikes (in thousands)')
plt.legend(['Dislikes'])

plt.show()




cor1=np.corrcoef(ca.views,ca.dislikes)
print("Correlation of Views and Dislikes: ")
print(cor1)


ca['likes_log'] = np.log(ca['likes'] + 1)
ca['views_log'] = np.log(ca['views'] + 1)
ca['dislikes_log'] = np.log(ca['dislikes'] + 1)
ca['comment_log'] = np.log(ca['comment_count'] + 1)

