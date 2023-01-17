import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame as show 

from os import path

from scipy.misc import imread

import random
df = pd.read_csv(r"../input/movie_metadata.csv")
df.head(5)
df.columns
plt.figure(figsize=(20,10))

print(df.isnull().sum())

plt.title('Checking for NaN values')

sns.heatmap(df.isnull(),annot=False,yticklabels=False,cbar=False,cmap='hot')
plt.figure(figsize=(15,10))

plt.subplots_adjust(hspace = .2)

plt.subplot(2,4,1)

sns.heatmap(df['gross'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,2)

sns.heatmap(df['budget'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,3)

sns.heatmap(df['content_rating'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,4)

sns.heatmap(df['aspect_ratio'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,5)

sns.heatmap(df['duration'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,6)

sns.heatmap(df['director_name'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,7)

sns.heatmap(df['color'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,8)

sns.heatmap(df['aspect_ratio'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.plot()

plt.plot()
df['color'] = df['color'].fillna(' Black and White')

df['duration'] = df['duration'].fillna(df['duration'].mean())

df['budget'] = df['budget'].fillna(df['budget'].mean())

df['gross'] = df['gross'].fillna(df['gross'].mean())

df['director_name'] = df['director_name'].fillna('unknown')

df['content_rating'] = df['content_rating'].fillna('PG-13')

df['aspect_ratio'] = df['aspect_ratio'].fillna(df['aspect_ratio'].mean())

df['title_year'] = df['title_year'].fillna(0)

df['country'] = df['country'].fillna('Not defined')

df['actor_1_facebook_likes'] = df['actor_1_facebook_likes'].fillna(0)

df['actor_2_facebook_likes'] = df['actor_2_facebook_likes'].fillna(0)

df['actor_3_facebook_likes'] = df['actor_3_facebook_likes'].fillna(0)

df['director_facebook_likes'] = df['director_facebook_likes'].fillna(0)

df['actor_1_name'] = df['actor_1_name'].fillna('Not defined')

df['actor_2_name'] = df['actor_2_name'].fillna('Not defined')

df['actor_3_name'] = df['actor_3_name'].fillna('Not defined')

df['language'] = df['language'].fillna('Not defined')

df['num_critic_for_reviews'] = df['num_critic_for_reviews'].fillna(df['num_critic_for_reviews'].mean())

plt.figure(figsize=(15,10))

plt.subplots_adjust(hspace = .2)

plt.subplot(2,4,1)

sns.heatmap(df['gross'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,2)

sns.heatmap(df['budget'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,3)

sns.heatmap(df['content_rating'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,4)

sns.heatmap(df['aspect_ratio'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='plasma')

plt.subplot(2,4,5)

sns.heatmap(df['duration'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,6)

sns.heatmap(df['director_name'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,7)

sns.heatmap(df['color'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,4,8)

sns.heatmap(df['aspect_ratio'].isnull().to_frame(),annot=False,yticklabels=False,cbar=False,cmap='viridis')

plt.plot()

plt.plot()
df.drop(labels='num_user_for_reviews',inplace = True,axis=1)
df.drop(labels='plot_keywords',inplace = True,axis=1)
df.drop(labels='facenumber_in_poster',inplace = True,axis=1)
plt.figure(figsize=(10,5))

plt.title('Checking for NaN values')

sns.heatmap(df.isnull(),annot=False,yticklabels=False,cbar=False,cmap='hot')
Imdb_Sc = df.sort_values(by=['imdb_score'],ascending=False).head(20)

plt.figure(figsize=(15,12))

plt.title('Top Movies according to IMDb',fontsize=20)

plt.xlabel('IMDb Score',fontsize=18)

plt.ylabel('Movie Title',fontsize=18)

bar = sns.barplot(y='movie_title',x='imdb_score',data=Imdb_Sc,palette='spring')
plt.figure(figsize=(15,4))

plt.subplots_adjust(hspace = .2)

plt.subplot(1,2,1)

bar = df['country'].value_counts().head(10).plot(kind = 'bar',cmap='summer')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)

plt.subplot(1,2,2)

df['country'].value_counts().head(10).plot(kind = 'pie',cmap='jet')

plt.show()
duration_df = df.sort_values(by=['duration'],ascending=False).head(10)

plt.figure(figsize=(20,8))

plt.title('Most Lenthy Movies',fontsize=20)

plt.xlabel('Movie Title',fontsize=18)

plt.ylabel('Length ( in Minutes )',fontsize=18)

bar = sns.barplot(x='movie_title',y='duration',data=duration_df,palette='hsv')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=18)

plt.show()
temp_df = df[df['director_name'] != 'unknown']
plt.figure(figsize = (15,6))

plt.title('Movies directed by most of the Directors')

bar = temp_df['director_name'].value_counts().head(20).plot(kind = 'bar')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)
plt.figure(figsize=(15,4))

plt.subplots_adjust(hspace = .2)

plt.subplot(1,2,1)

plt.title('Actor 1')

plt.xlabel('Actor 1 name ')

plt.xticks(np.arange(1, 81, 10))

df['actor_1_name'].value_counts().head(10).plot(kind = 'bar',colors = ['purple','indigo','skyblue','darkgreen'])

plt.subplot(1,2,2)

plt.title('Actor 2')

plt.xlabel('Actor 2 name ')

df['actor_2_name'].value_counts().head(10).plot(kind = 'bar',colors = ['skyblue','darkgreen','purple','indigo'])

plt.show()
Indian = df[df['country'] == 'India' ]

plt.figure(figsize=(15,6))

plt.title('Top Indian Directors',fontsize=14)

plt.xlabel('Names ',fontsize=14)

bar=Indian['director_name'].value_counts().head(10).plot(kind = 'bar')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)
plt.figure(figsize = (15,6))

temp_df = df[df['language'] != 'Not defined']

show(temp_df['language'].value_counts().head(10))
plt.figure(figsize=(10,6))

bar = temp_df['language'].value_counts().head(10).plot(kind = 'bar',colors = ['skyblue','darkgreen','purple','indigo','darkred'])

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)
hindi_df = df[df['language'] == 'Hindi']

show(hindi_df['movie_title'])
Imdb_Sc = hindi_df.sort_values(by=['imdb_score'],ascending=False).head(10)

plt.figure(figsize=(15,6))

plt.title('Top Movies according to IMDb',fontsize=20)

plt.xlabel('IMDb Score',fontsize=18)

plt.ylabel('Movie Title',fontsize=18)

bar = sns.barplot(x='movie_title',y='imdb_score',data=Imdb_Sc,palette='hsv')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)
like = hindi_df.sort_values(by=['movie_facebook_likes'],ascending=False).head(10)

plt.figure(figsize=(15,6))

plt.title('Top Movies according to people',fontsize=20)

plt.xlabel('Likes',fontsize=18)

plt.ylabel('Movie Title',fontsize=18)

bar = sns.barplot(x='movie_title',y='movie_facebook_likes',data=like,palette='gnuplot2')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70,fontsize=14)
plt.figure(figsize=(15,5))

Imdb_Sc = df.sort_values(by=['movie_facebook_likes'],ascending=False).head(20)

bar = sns.barplot(x='movie_title',y='movie_facebook_likes',data=Imdb_Sc,palette='hsv')

bar.set_xticklabels(bar.get_xticklabels(),rotation=70)