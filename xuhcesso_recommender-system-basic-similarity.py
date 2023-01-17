import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
plt.rcParams['patch.force_edgecolor']=True
df_features = pd.read_csv('../input/rating.csv', encoding="ISO-8859-1")
df_anime = pd.read_csv('../input/anime.csv', encoding="ISO-8859-1")
df_features.info()
df_features['rating'].describe(percentiles=[0.5])
df_features.head()
watched_animes = df_features.replace(to_replace=-1, value=np.nan)
watched_animes['rating'].describe(percentiles=[0.5])
n_user = df_features['user_id'].nunique()
n_anime = df_features['anime_id'].nunique()
print('Originally we had {0} users and {1} animes'.format(n_user, n_anime))
print('After dropping N/A values we have {0} users and {1} animes'.format(watched_animes.dropna()['user_id'].nunique(), watched_animes.dropna()['anime_id'].nunique()))
sns.distplot(watched_animes['rating'].dropna(), color='red')
df_anime.head()
drop_anime = watched_animes['anime_id'][watched_animes['rating'].isnull()]
df_anime = df_anime.drop(df_anime['anime_id'].isin(drop_anime))
print("We have {0} in the ratings database and {1} in the anime database.".format(n_anime, df_anime['anime_id'].count()))
sns.countplot(x='type', data=df_anime)
sns.distplot(df_anime['rating'].dropna(), color='red')
plt.figure(figsize=(10,6))
sns.countplot(df_anime[df_anime['type']=='TV']['episodes'].value_counts(), palette='viridis')
plt.tight_layout()
plt.figure(figsize=(10,6))
sns.jointplot(x='rating', y='members', data=df_anime, s=10, color='green')
genre = df_anime.genre.str.get_dummies(sep=",")
print('We have {0} genres in the database'.format(genre.shape[1]))
sum_genre = pd.DataFrame(genre.sum(), index=genre.columns, columns=['Total'])
plt.figure(figsize=(14,8))
sns.barplot(data=sum_genre.sort_values(by=['Total'],ascending=False).head(15).T)
plt.tight_layout()
names = df_anime[['name', 'anime_id']]
watched_animes = pd.merge(names, watched_animes, on='anime_id')
watched_animes.drop('anime_id', inplace=True,axis=1)
ratings = pd.DataFrame(watched_animes.groupby('name')['rating'].mean())
ratings['Number of Ratings'] = pd.DataFrame(watched_animes.groupby('name')['rating'].count())
plt.figure(figsize=(10,6))
sns.distplot(ratings['Number of Ratings'], kde=False, bins=250, color=sns.color_palette('rocket',1)[0])
plt.xlim((0,5000))
user_item = watched_animes.sample(30000)
user_item = user_item.pivot_table(values='rating', index=['user_id'], columns=['name'])
user_item.fillna(value=0, inplace=True)
fullmetal = user_item['Fullmetal Alchemist']
similar_fullmetal = user_item.corrwith(fullmetal)
corr_fullmetal = pd.DataFrame(similar_fullmetal, columns=['Correlation'])
corr_fullmetal = corr_fullmetal.join(ratings['Number of Ratings'])
corr_fullmetal = corr_fullmetal[corr_fullmetal['Number of Ratings']>1000].sort_values('Correlation', ascending=False).head()
animes = corr_fullmetal.index
df = df_anime.set_index('name')
for name in animes[0:3]:
    fm = df.loc[name]
    print("{0}'s genre is: {1}. It is a {2} anime with {3} episodes and {4} members. The rating is {5}\n".format(name,
      fm['genre'], fm['type'], fm['episodes'], fm['members'], fm['rating']))
from sklearn.metrics.pairwise import cosine_similarity
similar_fullmetal = cosine_similarity(user_item.T,fullmetal.values.reshape(-1,fullmetal.shape[0]))
corr_fullmetal = pd.DataFrame(similar_fullmetal, columns=['Correlation'], index=user_item.columns)
corr_fullmetal = corr_fullmetal.join(ratings['Number of Ratings'])
corr_fullmetal = corr_fullmetal[corr_fullmetal['Number of Ratings']>1000].sort_values('Correlation', ascending=False).head()
animes = corr_fullmetal.index
df = df_anime.set_index('name')
for name in animes[0:3]:
    fm = df.loc[name]
    print("{0}'s genre is: {1}. It is a {2} anime with {3} episodes and {4} members. The rating is {5}\n".format(name,
      fm['genre'], fm['type'], fm['episodes'], fm['members'], fm['rating']))
one = user_item['One Punch Man']
similar_one = user_item.corrwith(one)
corr_one = pd.DataFrame(similar_one, columns=['Correlation'])
corr_one = corr_one.join(ratings['Number of Ratings'])
corr_one = corr_one[corr_one['Number of Ratings']>1000].sort_values('Correlation', ascending=False).head()
animes = corr_one.index
for name in animes[0:3]:
    fm = df.loc[name]
    print("{0}'s genre is: {1}. It is a {2} anime with {3} episodes and {4} members. The rating is {5}\n".format(name,
      fm['genre'], fm['type'], fm['episodes'], fm['members'], fm['rating']))
one = user_item['One Punch Man']
similar_one = cosine_similarity(user_item.T,one.values.reshape(-1,one.shape[0]))
corr_one = pd.DataFrame(similar_one, columns=['Correlation'], index=user_item.columns)
corr_one = corr_one.join(ratings['Number of Ratings'])
corr_one = corr_one[corr_one['Number of Ratings']>1000].sort_values('Correlation', ascending=False).head()
animes = corr_one.index
for name in animes[0:3]:
    fm = df.loc[name]
    print("{0}'s genre is: {1}. It is a {2} anime with {3} episodes and {4} members. The rating is {5}\n".format(name,
      fm['genre'], fm['type'], fm['episodes'], fm['members'], fm['rating']))