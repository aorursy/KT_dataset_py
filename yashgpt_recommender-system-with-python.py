import numpy as np

import pandas as pd
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('../input/u.csv', sep='\t', names=column_names)
df.head()
movie_titles = pd.read_csv("../input/Movie_Id_Titles.csv")

movie_titles.head()
df = pd.merge(df,movie_titles,on='item_id')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

%matplotlib inline
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings.head()
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

ratings.head()
plt.figure(figsize=(10,4))

ratings['num of ratings'].hist(bins=70)
plt.figure(figsize=(10,4))

ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

moviemat.head()
ratings.sort_values('num of ratings',ascending=False).head(10)
ratings.head()
starwars_user_ratings = moviemat['Star Wars (1977)']

toystory_user_ratings = moviemat['Toy Story (1995)']

starwars_user_ratings.head()
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

similar_to_toystory = moviemat.corrwith(toystory_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])

corr_starwars.dropna(inplace=True)

corr_starwars.head()
corr_starwars.sort_values('Correlation',ascending=False).head(10)
corr_starwars = corr_starwars.join(ratings['num of ratings'])

corr_starwars.head()
corr_starwars[corr_starwars['num of ratings']>200].sort_values('Correlation',ascending=False).head()
corr_toystory = pd.DataFrame(similar_to_toystory,columns=['Correlation'])

corr_toystory.dropna(inplace=True)

corr_toystory = corr_toystory.join(ratings['num of ratings'])

corr_toystory[corr_toystory['num of ratings']>200].sort_values('Correlation',ascending=False).head()