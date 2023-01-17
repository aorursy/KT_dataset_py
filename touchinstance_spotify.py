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
df = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data.csv")

print(df.shape)

df.columns
df.isna().sum().sum()
df.drop([

'Unnamed: 0', 'id','explicit','key','release_date','mode'], axis=1, inplace=True

)

df.head()
# 查看column之间的相关性

corr = df[['acousticness','danceability','energy',

'instrumentalness','liveness','tempo','valence']].corr()

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

%matplotlib inline

plt.figure(figsize=(12,8))

sns.heatmap(corr, annot=True)
df[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy', ascending=False)[:10]

# 越积极向上的歌曲，不插电的程度越低

df.acousticness.mean()

# 以时间为维度，看一下歌曲各项指标的发展趋势

year_avg = df[['danceability','energy','liveness','acousticness', 'valence','year']].groupby('year').mean().sort_values(by='year').reset_index()

year_avg.head()
plt.figure(figsize=(14,8))

plt.title("Song Trends Over Time", fontsize=15)

lines = ['danceability','energy','liveness','acousticness','valence']

for line in lines:

    ax = sns.lineplot(x='year', y=line, data=year_avg)

plt.legend(lines)
# 使用melt固定year这一列，同时行转列打散其他列

melted = year_avg.melt(id_vars='year')

melted.head()

plt.figure(figsize=(14,6))

plt.title("Song Trends Over Time", fontsize=15)

sns.lineplot(x='year', y='value', hue='variable', data=melted)
# 总共的歌手人数

df.artists.nunique()

# value_counts() 是查看每列有多少不同的值，以及每个值有多少重复的次数

df.artists.value_counts()[:7]

artist_list = df.artists.value_counts().index[:7]

artist_list
df_artists = df[df.artists.isin(artist_list)][['artists','year',

                                                        'energy']].groupby(['artists','year']).count().reset_index()

df_artists.rename(columns={'energy':'song_count'}, inplace=True)

df_artists.head()
plt.figure(figsize=(16,8))

sns.lineplot(x='year', y='song_count', hue='artists', data=df_artists)
df1 = pd.DataFrame(np.zeros((100,7)), columns=artist_list)

df1['year'] = np.arange(1921,2021)

print(df1.shape)

df1.head()
df1 = df1.melt(id_vars='year',var_name='artists', value_name='song_count')

print(df1.shape)

df1.head()
df_merge = pd.merge(df1, df_artists, on=['year','artists'], how='outer').sort_values(by='year').reset_index(drop=True)

df_merge.head()
#  inplace = True：不创建新的对象，直接对原始对象进行修改；

df_merge.fillna(0, inplace=True)

df_merge.drop('song_count_x', axis=1, inplace=True)

df_merge.rename(columns={'song_count_y':'song_count'}, inplace=True)

df_merge.head()
# cumsum 累加计算

df_merge['cumsum'] = df_merge[['song_count','artists']].groupby('artists').cumsum()

df_merge.head(10)
import plotly.express as px

fig = px.bar(df_merge,

             x='artists', y='cumsum',

             color='artists',

             animation_frame='year', animation_group='year',

             range_y=[0,1000],

             title='Artists with Most Number of Songs')

fig.show()