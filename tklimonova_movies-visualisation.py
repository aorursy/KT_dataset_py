import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
imdb_filepath = "../input/imdb-data/IMDB-Movie-Data.csv"

imdb_data = pd.read_csv(imdb_filepath, index_col='Rank')

imdb_data.head()
#to see how many columns and rows in the df

imdb_data.shape
#Top 20 movies with the highest rating

most_rated = imdb_data.sort_values('Rating', ascending = False).head(20).set_index('Title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['Rating'], most_rated.index, palette='deep')

plt.title("Top 20 movies with the highest rating", fontsize=18)

plt.xlabel("Movie rating")

plt.ylabel("Movie title")

plt.show()
#Top 20 movies with the highest revenue

highest_revenue = imdb_data.sort_values('Revenue (Millions)', ascending = False).head(20).set_index('Title')

plt.figure(figsize=(12,10))

sns.barplot(highest_revenue['Revenue (Millions)'], highest_revenue.index, palette='CMRmap')

plt.title("Top 20 movies with the highest revenue", fontsize=18)

plt.xlabel("Movie revenue")

plt.ylabel("Movie title")

plt.show()
#Revenue of all the movies per year

df = imdb_data.groupby('Year')['Revenue (Millions)'].sum().to_frame().reset_index().sort_values(by='Revenue (Millions)')



plt.figure(figsize=(12,10))

sns.barplot(df['Year'], df['Revenue (Millions)'], palette='rocket')

plt.title("Revenue per year of all movies", fontsize=18)

plt.xlabel("Year movie was released")

plt.ylabel("Movies revenue")

plt.show()
#The top 10 highly rated Directors

high_rated_director = imdb_data[imdb_data['Rating']>=7.5]

high_rated_director = high_rated_director.groupby('Director')['Title'].count().reset_index().sort_values('Title', ascending = False).head(10).set_index('Director')



plt.figure(figsize=(15,10))

ax = sns.barplot(high_rated_director['Title'], high_rated_director.index, palette='Set2')

plt.title("Top 10 highly rated Directors", fontsize=18)

plt.xlabel("Number of Movies")

plt.ylabel("Director")

plt.show()
#Corelation between revenue and rating of the movies

sns.scatterplot(x=imdb_data['Revenue (Millions)'], y=imdb_data['Rating'], hue = imdb_data['Votes'])

plt.show()
#Corelation between Metascore and rating of the movies

sns.scatterplot(x=imdb_data['Metascore'], y=imdb_data['Rating'], hue = imdb_data['Votes'])

plt.show()
#Total amount of movies per year

df = imdb_data.groupby('Year')['Title'].count().to_frame().reset_index().sort_values(by='Year')



plt.figure(figsize=(12,10))

sns.barplot(df['Year'], df['Title'], palette='winter')

plt.title("Revenue per year of all movies", fontsize=18)

plt.xlabel("Year movie was released")

plt.ylabel("Movies revenue")

plt.show()
#what is the most common Rating for movies in this dataset?

plt.figure(figsize=(10,10))

rating= imdb_data.Rating.astype(float)

sns.distplot(rating, bins=20)

plt.show()