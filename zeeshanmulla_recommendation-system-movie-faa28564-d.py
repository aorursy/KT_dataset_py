import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/Recommendation System.csv')
df.head()
df.columns = (['user_id','item_id',"rating","timestamp"])
df.head()
movie_titles = pd.read_csv("../input/Movie_Id_Titles")

movie_titles.head()
df = pd.merge(df,movie_titles,on='item_id')
df.sort_values('item_id').head()
df.head()
df.groupby("title")['rating'].mean().sort_values(ascending=False).head()
df.groupby("title")['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby("title")['rating'].mean())

ratings.head()
ratings["rating_counts"]= pd.DataFrame(df.groupby("title")['rating'].count())
ratings.head()
sns.distplot(ratings["rating_counts"])
# plt.figure(figsize=(20,10))

sns.distplot(ratings["rating"],bins=50)
sns.distplot( (ratings['rating']*ratings['rating_counts'] ) )
plt.figure(figsize=(10,7))

sns.jointplot(x='rating',y="rating_counts",data=ratings,alpha=.5)
df.head()
movie_mat = df.pivot_table(values='rating',index='user_id',columns = 'title')
movie_mat.head()
ratings.sort_values("rating_counts",ascending=False).head()
star_war_user_ratin = movie_mat['Star Wars (1977)']

star_war_user_ratin.value_counts()
liar_liar_user_ratin = movie_mat['Liar Liar (1997)']

liar_liar_user_ratin.head()
liar_liar_user_ratin.value_counts()
similar_to_star_wars = movie_mat.corrwith(star_war_user_ratin)

similar_to_star_wars.head()
similar_to_liarliar = movie_mat.corrwith(liar_liar_user_ratin)

similar_to_liarliar.head()
corr_star_wards = pd.DataFrame(similar_to_star_wars,columns=['Correlation'])
corr_star_wards.dropna(inplace=True)
corr_star_wards.sort_values('Correlation',ascending=False).head()
corr_star_wards= corr_star_wards.join(ratings["rating_counts"])
corr_star_wards.head()
( corr_star_wards[corr_star_wards['rating_counts']>100] ).sort_values('Correlation',ascending=False).head()
# Finding movies similar to Liar
similar_to_liarliar.head()
corr_liarliar= pd.DataFrame(similar_to_liarliar,columns=["Correlation"])

corr_liarliar.dropna(inplace=True)

corr_liarliar.head()
corr_liarliar.sort_values("Correlation",ascending=False).head()
corr_liarliar=corr_liarliar.join(ratings["rating_counts"])



corr_liarliar.head()
corr_liarliar[corr_liarliar["rating_counts"]>100].sort_values('Correlation',ascending=False)