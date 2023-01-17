import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import STOPWORDS,WordCloud
# Loading USVideos and Comments only

df_videos = pd.read_csv('../input/USvideos.csv',error_bad_lines=False)

df_comments = pd.read_csv('../input/UScomments.csv',error_bad_lines=False)
df_videos_top = df_videos[df_videos['tags'].str.contains('iPhone' or 'Apple')].sort_values(by='views', ascending=False).head(10)

ax = df_videos_top.plot(kind='bar',x='channel_title',y='views',title='Top10 iPhone channels')

ax.set_xlabel('Channel Title')

ax.set_ylabel('Views(in m)')
cat_id_mapping = {2:'Autos & Vehicles',1:'Film & Animation',

                  10:'Music',15:'Pets & Animals',17:'Sports',

                 19:'Travel & Events',20:'Gaming',22:'People & Blogs',

                 23:'Comedy',24:'Entertainment',25:'News & Politics',

                 26:'Howto & Style',27:'Education',28:'Science & Technology',

                  29:'Nonprofits & Activism',43:'Shows'}

df_videos_gb = df_videos.groupby('category_id').count()['title']

df_videos_gb = df_videos_gb.rename(cat_id_mapping)

ax = df_videos_gb.plot(kind='bar',title='Video Categories by their Count',color='green',figsize=(10,5))

ax.set_xlabel('Category')

ax.set_ylabel('Count')
sns.pairplot(df_videos,x_vars=['comment_total','views'],y_vars=['likes','dislikes'],size=5)
sns.jointplot(x='views',y='likes',data=df_videos,kind='reg')
df_combined = pd.merge(df_videos,df_comments,on='video_id')

df_combined = df_combined.groupby('title').count().sort_values(by='comment_text',ascending=False).head(20)

sns.set(font_scale=1.5)

ax = sns.barplot(x=df_combined.index,y='comment_text',data=df_combined)

ax.set_xticklabels(labels=df_combined.index,rotation=90)

ax.set_xlabel('Title')

ax.set_ylabel('Count')
df_videos.head()

tags = df_videos['tags'].map(lambda x:x.lower().split('|')).values

splt_tags = ' '.join(df_videos['tags'])

wordcloud = WordCloud(width=1000,height=500).generate(' '.join(splt_tags.lower().split('|')))



plt.figure(figsize=(15,5))

plt.imshow(wordcloud)

plt.axis('off')