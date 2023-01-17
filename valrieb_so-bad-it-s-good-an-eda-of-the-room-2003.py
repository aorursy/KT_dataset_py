#importing the python libraries we'll need



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import scipy.stats as ss



#ingesting the two datasets



movies=pd.read_csv('../input/movielens-25m/ml-25m/movies.csv')

ratings=pd.read_csv('../input/movielens-25m/ml-25m/ratings.csv')
#a quick look at the "movies" dataframe and verifying if there's any NaN value



print(movies.isnull().any())

print(movies.shape)

movies.head()
#a quick look at the "ratings" dataframe and verifying if there's any NaN value



print(ratings.isnull().any())

print(ratings.shape)

ratings.head()
#merging the two DataFrames 



movies = ratings.merge(movies, on='movieId', how='inner')

movies.head()
#converting the unix timestamps into readable dates



movies['date'] = pd.to_datetime(movies['timestamp'], unit='s')

movies.head()
#creating a new 'year' column by extracting the year from the 'date' column



movies['year'] = pd.DatetimeIndex(movies['date']).year

movies.head()
#deleting the columns we won't need for our study



del movies['genres']

del movies['userId']

del movies['timestamp']

del movies['date']

movies.head()
#locating "The Room" in the DataFrame



is_theroom = (movies['title'].str.contains('Room')) & (movies['title'].str.contains('2003'))

theroom = movies[is_theroom]

theroom.head()
#transforming the DataFrame to only keep "The Room"'s data



is_theroom = movies['movieId'] == 74754

theroom= movies[is_theroom]

theroom.head()
print('How many ratings do we have for The Room?:',len(theroom))
years = theroom['year'].unique().tolist()

print('And for how many years?:', len(years))
theroom['rating'].describe()
#computing "The GodFather" stats - its movieId is 858



is_thegodfather = movies['movieId'] == 858

thegodfather= movies[is_thegodfather]

thegodfather['rating'].describe()
#computing "The Last Airbender" stats - its movieId is 78893



is_thelastairbender = movies['movieId'] == 78893

thelastairbender= movies[is_thelastairbender]

thelastairbender['rating'].describe()
#computing the pdf of the 3 movies



x = np.linspace(-5, 10, 5000)

room_pdf = ss.norm.pdf(x, 2.403731, 1.664965) 

godfather_pdf = ss.norm.pdf(x, 4.324336, 0.873055) 

airbender_pdf = ss.norm.pdf(x, 2.006278, 1.248413) 



#creating the plot



figure(figsize=(13, 8))

plt.title('Gaussian Distributions', size=20, color='firebrick')

plt.plot(x, room_pdf, color='firebrick', label='The Room')

plt.plot(x, godfather_pdf, color='goldenrod', label='The Godfather')

plt.plot(x, airbender_pdf, color='yellowgreen', label='The Last Airbender')

plt.legend();
#deleting more columns: we only need the ratings and the years.



del theroom['title']

del theroom['movieId']
#counting the number of ratings per year



ratings_counts = theroom['year'].value_counts().sort_index()

ratings_counts
#drawing the plot



ratings_counts.plot(kind='bar', figsize=(13,8), color='yellowgreen', grid=True)

plt.xlabel('years')

plt.ylabel('number of ratings')

plt.title('Number of Ratings per year', size=20, color='chocolate');
#computing the average rating per year



theroomav = theroom[['year','rating']].groupby('year', as_index=False).mean()

theroomav
#creating the plot



theroomav.plot(x='year', y='rating', figsize=(13,8), grid=True, color='firebrick')

plt.xlabel('years')

plt.ylabel('ratings')

plt.axis([2010, 2019,0,5])

plt.title('Evolution of the Average Rating', size=20, color='goldenrod');