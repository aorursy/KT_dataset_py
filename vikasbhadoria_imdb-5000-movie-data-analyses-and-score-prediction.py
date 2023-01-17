import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None) 
#Loading the dataset
df = pd.read_csv('../input/imdb-5000-movie-dataset/movie_metadata.csv')
df.sample(5)
df.columns,df.shape
plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(),cmap='Blues',cbar=False,yticklabels=False)
# Movie with the lowest Imdb rating is Documentary `Justin Bieber: Never Say Never`
df_low_imdb=df[df['imdb_score']==1.6]
df_low_imdb
# Movie with the highest Imdb rating is Comedy `Towering Inferno`
df_max_imdb=df[df['imdb_score']==9.5]
df_max_imdb
df.hist(bins=30,figsize=(15,15),color='g')
df['genres_num'] = df.genres.apply(lambda x: len(x.split('|')))
df.head(2)
df['genres_num'].max()
df[df['genres_num']==8]
df['Type_of_genres'] = df.genres.apply(lambda x: x.replace('|',','))
df.head(2)
df['genres_first'] = df.genres.apply(lambda x: x.split('|')[0] if '|' in x else x)
df.head()
plt.figure(figsize=(14,10))
sns.boxplot(x='imdb_score',y='genres_first',data=df)
correlations = df.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)
df_for_ML = df[['num_critic_for_reviews','duration','director_facebook_likes','num_voted_users','num_user_for_reviews']]
pd.pivot_table(df,index='genres_first', values='imdb_score').sort_values('imdb_score', ascending = False)
# There seems some outliers here. As we saw clearly above in our boxplot that Documentary are highest rated on an average. 
df.title_year.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(20,20),color='g')
df.genres_first.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(10,10),color='g')
df.country.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(20,20),color='g')
df_for_ML.head(2)
for i in df_for_ML.columns:
    axis = df.groupby('imdb_score')[[i]].mean().plot(figsize=(10,5),marker='o',color='g')
df_for_ML["num_critic_for_reviews"] = df_for_ML["num_critic_for_reviews"].fillna(df["num_critic_for_reviews"].mean())
df_for_ML["duration"] = df_for_ML["duration"].fillna(df["duration"].mean())
df_for_ML["director_facebook_likes"] = df_for_ML["director_facebook_likes"].fillna(df["director_facebook_likes"].mean())
df_for_ML["num_user_for_reviews"] = df_for_ML["num_user_for_reviews"].fillna(df["num_user_for_reviews"].mean())
df_for_ML["num_voted_users"] = df_for_ML["num_voted_users"].fillna(df["num_voted_users"].mean())
sns.heatmap(df_for_ML.isnull(),cmap='Blues',cbar=False,yticklabels=False)
df_for_ML.info()
from sklearn.model_selection import train_test_split
X = df_for_ML
y = df['imdb_score']
X.shape,y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train,y_train)
prec_lm=lm.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('The mean squared error using Linear regression is: ',mean_squared_error(y_test,prec_lm))
print('The mean absolute error using Linear regression is: ',mean_absolute_error(y_test,prec_lm))
### Using the Xgboost model for predicting the the imdb score.
from xgboost import XGBClassifier
Xgb = XGBClassifier()
Xgb.fit(X_train,y_train)
prec_Xgb=Xgb.predict(X_test)
print('The mean squared error using the Xgboost model is: ',mean_squared_error(y_test,prec_Xgb))
print('The mean absolute error using the Xgboost model is: ',mean_absolute_error(y_test,prec_Xgb))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
prec_rf=rf.predict(X_test)
print('The mean squared error using Random Forest model is: ',mean_squared_error(y_test,prec_rf))
print('The mean absolute error using Random Forest model is: ',mean_absolute_error(y_test,prec_rf))
