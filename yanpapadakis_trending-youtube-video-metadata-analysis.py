# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime as dt

# Import US Data
us_videos = pd.read_csv('../input/USvideos.csv')
us_videos_categories = pd.read_json('../input/US_category_id.json')

# Map Category IDs using the supporting file: US_category_id.json
categories = {int(category['id']): category['snippet']['title'] for category in us_videos_categories['items']}
# First Row
us_videos.head(5)
# All entries for first-row video
us_videos[us_videos.video_id == "2kyS6SvSYSE"]
# Category ID will be used to assign categories later, it is not a numeric variable.
us_videos.category_id = us_videos.category_id.astype('category')

# Get Metadata Information
us_videos.info()
# Summary of Object Variables
us_videos.describe(include=[np.object])
# Most Frequent Category, 24 is
print(categories[24])
# Summary of Boolean Variables
us_videos.select_dtypes(include=[np.bool]).apply(pd.Series.value_counts,dropna=False)
# Summary of Numeric Variables
us_videos.describe(percentiles=[.05,.25,.5,.75,.95]).round(1)
counter = 0
for k,v in categories.items():
    print('{:2d}: {:24}'.format(k,v),end=' ')
    counter += 1
    if counter % 4 == 0:
        print()
print('\n{} Categories in Total.'.format(counter))
# Transform trending_date to datetime date format
us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'], format='%y.%d.%m').dt.date
us_videos.trending_date.value_counts().sort_index(inplace=True)
us_videos.head()
# Dataset is sorted by trending_date
pd.Index(us_videos.trending_date).is_monotonic
# Transforming publish_time to datetime
publish_time = pd.to_datetime(us_videos.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')

# Create Variable publish_date
us_videos['publish_date'] = publish_time.dt.date

# Drop publish_time
us_videos.drop('publish_time',axis=1,inplace=True)
# Create New Variable Counting Days to Achieving Trending Status
us_videos['days_to_trending'] = (us_videos.trending_date - us_videos.publish_date).dt.days
us_videos.days_to_trending.describe(percentiles=[.05,.25,.5,.75,.95])
us_videos.set_index(['trending_date','video_id'],inplace=True)
us_videos.head()
us_videos['dislike_percentage'] = us_videos['dislikes'] / (us_videos['dislikes'] + us_videos['likes'])
us_videos.dislike_percentage.describe(percentiles=[.05,.25,.5,.75,.95])
# how should we interpret 'video_error_or_removed' == True ?
print(us_videos[us_videos.video_error_or_removed])
us_videos = us_videos[~us_videos.video_error_or_removed]
# Video Level Stats Using First Occurence Values
video_level = us_videos.groupby(level=1).first()
video_level['freq'] = us_videos['title'].groupby(level=1).count()
video_level['category'] = [categories[cid] for cid in video_level.category_id]
video_level.drop('category_id',axis=1,inplace=True)
video_level.sort_values(by=['views'],ascending=False,inplace=True)
video_level['views_ratio'] = us_videos['views'].groupby(level=1).last() / video_level.views
views_min_dt = pd.Series([t[0] for t in us_videos['views'].groupby(level=1).idxmin()],index=video_level.index)
video_level['views_min_dt'] = views_min_dt
video_level.head(10)
video_level.describe(percentiles=[.05,.25,.5,.75,.95])
from IPython.display import HTML, display

# First Occurrence of the 10 longest lasting videos by days on list and first views count
tmp = video_level.sort_values(by=['freq','views'],ascending=False).head(10)
#
# Construction of HTML table with miniature photos assigned to the most popular videos
table = '<h1>Trending the Longest by Days Trending and Initial Views</h1><table>'

# Add Header
table += '<tr>'
table += '<th>Photo</th><th>Channel Name</th><th style="width:250px;">Title</th><th>Category</th><th>Publish Date</th>'
table += '<th>Days Trending</th><th>Views</th>'
table += '</tr>'

max_title_length = 50

for video_id, row in tmp.iterrows():
    table += '<tr>'
    table += '<td><img src="{thumbnail_link}" style="width:100px;height:100px;"></td>'.format(**row)
    table += '<td>{channel_title}</td>'.format(**row)
    table += '<td>{title}</td>'.format(**row)
    table += '<td>{category}</td>'.format(**row)
    table += '<td>{publish_date}</td>'.format(**row)
    table += '<td>{freq}</td>'.format(**row)
    table += '<td align="right">{views:11,d}</td>'.format(**row)
    table += '</tr>'  
table += '</table>'

display(HTML(table))
tmp = video_level[['freq','days_to_trending']]
days_to_trending_max = us_videos.groupby(level=1)[['days_to_trending']].max()
tmp = tmp.join(days_to_trending_max,how='left',rsuffix='_max')
tmp['test'] = tmp.days_to_trending_max - tmp.days_to_trending + 1
print('{:.2%}'.format(sum([a==b for a,b in zip(tmp.freq,tmp.test)]) / len(tmp.index)))
tmp[tmp.test != tmp.freq].head()
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
tmp = us_videos.channel_title.value_counts()[:25]
_ = sns.barplot(y=tmp.index,x=tmp)
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
sns_ax = sns.countplot([categories[i] for i in us_videos.category_id])
_, labels = plt.xticks()
_ = sns_ax.set_xticklabels(labels, rotation=60)
table = pd.pivot_table(us_videos, index=us_videos.index.labels[0])
table.index = us_videos.index.levels[0]
_ = table[['likes','dislikes','comment_count']].plot()
_ = table[['views']].plot()
_ = table[['comments_disabled','ratings_disabled','video_error_or_removed']].plot()
max_days_to_trending = us_videos.sample(1000).groupby('video_id').days_to_trending.max() # Notice Sampling: EDA Principle 3
sns_ax = sns.boxplot(y=max_days_to_trending)
_ = sns_ax.set(yscale="log")
plt.show()
_ = sns.distplot(max_days_to_trending.value_counts(),bins='rice',kde=False)
sns_ax = sns.distplot(np.nan_to_num(us_videos.sample(1000).dislike_percentage),bins='fd')  # Notice Sampling: EDA Principle 3
_ = sns_ax.set_title('Distribution of Dislike Percentage')
plt.figure(figsize=(16,8))
plt.title('Not Instant Hits: Success by Dislike Percentage', fontsize=20)
plt.xlabel('Likes', fontsize=15)
plt.ylabel('Dislike Percentage', fontsize=15)

min_days_to_trending = us_videos.groupby(level=1).days_to_trending.min()
videos_passing_test = us_videos.index.get_level_values('video_id').isin(min_days_to_trending[min_days_to_trending > 1].index)
tmp = us_videos[(videos_passing_test) & (us_videos.likes > 1000)].sample(1000)
plt.scatter(tmp.likes, tmp.dislike_percentage, c=tmp.category_id)
plt.show()

plt.figure(figsize=(16,8))
plt.title('Instant Hits: Success by Dislike Percentage', fontsize=20)
plt.xlabel('Likes', fontsize=15)
plt.ylabel('Dislike Percentage', fontsize=15)
tmp = us_videos[us_videos.index.get_level_values('video_id'). \
                isin(min_days_to_trending[min_days_to_trending <= 1].index) & (us_videos.likes > 1000)].sample(1000)
plt.scatter(tmp.likes, tmp.dislike_percentage, c=tmp.category_id)
plt.show()
tmp = video_level.sample(1000)[['days_to_trending','dislike_percentage','likes','dislikes','views','views_ratio','freq']] # Notice Sampling: EDA Principle 3
tmp = tmp[tmp.days_to_trending < 20]
tmp['log_views'] = np.log10(tmp.views)
tmp['trans_dislike_pct'] = np.log(tmp.likes+1) - np.log(tmp.dislikes+1)
tmp['log_views_ratio'] = np.log(tmp.views_ratio)
_ = sns.pairplot(tmp[['days_to_trending','freq','log_views','log_views_ratio','trans_dislike_pct']],dropna=True)
sns_ax = sns.boxplot(x='category',y='views',data=video_level)
_, labels = plt.xticks()
_ = sns_ax.set_xticklabels(labels, rotation=90)
_ = sns_ax.set_title('Views')
_ = sns_ax.set(yscale="log")
sns_ax = sns.boxplot(x='category',y='views_ratio',data=video_level)
_, labels = plt.xticks()
_ = sns_ax.set_xticklabels(labels, rotation=90)
_ = sns_ax.set_title('Views Ratio')
_ = sns_ax.set(yscale="log")
video_level['views_min_dt'] = video_level.views_min_dt.dt.to_period('Q')
tmp = video_level.groupby(['category','views_min_dt']).views_ratio.median()
_ = tmp.unstack().plot(kind='bar')
tmp = video_level[(video_level.views_ratio < 10) & (video_level.freq > 1) & (video_level.views_ratio > .8)].dropna().sort_values(by='views_ratio')
cat_ratio_median = tmp.groupby('category')['views_ratio'].median()
tmp = tmp.merge(cat_ratio_median.rename('cat_ratio_median').to_frame(), left_on='category',right_index=True)

y = np.log(tmp.views_ratio)
print('y')
print(y.describe(percentiles=[.05,.25,.5,.75,.95]))
X = tmp[['views','likes','dislikes','comment_count','dislike_percentage','days_to_trending','cat_ratio_median']]


tmp_log = np.log(tmp[['views','likes','dislikes','comment_count','dislike_percentage','days_to_trending','cat_ratio_median','views_ratio']]+1)
X_reg = tmp_log[['views','likes','dislikes','comment_count','dislike_percentage','days_to_trending','cat_ratio_median']]
y_reg = tmp_log.views_ratio
print('y_reg')
print(y_reg.describe(percentiles=[.05,.25,.5,.75,.95]))
_ = sns.pairplot(pd.concat((X,y.rename('y')),axis=1).sample(500)) # Notice Sampling: EDA Principle 3
_ = sns.pairplot(tmp_log.sample(500))  # Notice Sampling: EDA Principle 3
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lars, Ridge


regr_1 = DecisionTreeRegressor(max_depth=2,min_samples_leaf=.01)
regr_2 = DecisionTreeRegressor(max_depth=4,min_samples_leaf=.01)
regr_ols = LinearRegression()
regr_lrs = Lars()
regr_rdg = Ridge (alpha = .5)

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_ols.fit(X_reg, y_reg)
regr_lrs.fit(X_reg, y_reg)
regr_rdg.fit(X_reg, y_reg)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)
y_ols = regr_ols.predict(X_reg)
y_lrs = regr_lrs.predict(X_reg)
y_rdg = regr_rdg.predict(X_reg)

print((y_1 - y).describe(percentiles=[.05,.25,.5,.75,.95]))
print('Regression Tree 1: R-sq = {:.2%}'.format(metrics.r2_score(y,y_1)))


print((y_ols - y).describe(percentiles=[.05,.25,.5,.75,.95]))
print('OLS: R-sq = {:.2%}'.format(metrics.r2_score(y_reg,y_ols)))



print((y_lrs - y).describe(percentiles=[.05,.25,.5,.75,.95]))
print('LARS: R-sq = {:.2%}'.format(metrics.r2_score(y_reg,y_lrs)))



print((y_rdg - y).describe(percentiles=[.05,.25,.5,.75,.95]))
print('Ridge Regression: R-sq = {:.2%}'.format(metrics.r2_score(y_reg,y_rdg)))
import graphviz 
dot_data = tree.export_graphviz(regr_1, out_file=None, feature_names=X.columns) 
graphviz.Source(dot_data)
# Alternative Regression Tree with Bigger Depth
# Better Performance but More Difficult to Understand
print((y_2 - y).describe())
print('R-sq = {:.2%}'.format(metrics.r2_score(y,y_2)))
y_2_unique = pd.Series(pd.unique(y_2))
#print(pd.concat([y_2_unique.rename('y'),np.exp(y_2_unique).rename('exp(y)')],axis=1))
dot_data = tree.export_graphviz(regr_2, out_file=None, feature_names=X.columns) 
graphviz.Source(dot_data)
days_to_trending_cat = pd.cut(video_level.days_to_trending,[0,1,2,3,np.inf])
ax = video_level.groupby(days_to_trending_cat)['views_ratio'].median().plot(kind='bar')
_ = ax.set(ylabel="Median Views Ratio",xlabel="Days to Trending Groups")
# Categories with High Median Views Ratio
cat_ratio_median[cat_ratio_median > 1.837]