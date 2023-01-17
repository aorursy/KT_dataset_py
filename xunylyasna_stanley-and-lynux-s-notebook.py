import pandas as pd
import os

os.listdir('../input')
reviews = pd.read_csv('../input/reviews.csv')
genres = pd.read_csv('../input/genres.csv')
artists = pd.read_csv('../input/artists.csv')
genres.head()
df = pd.merge(reviews, genres, on='reviewid', how='outer')
df = pd.merge(df, artists, on='reviewid', how='outer')
df.head()
df.isnull().sum()
df = df.dropna(subset=['genre'])
#we drop entries without genre
df.isnull().sum()
print(df)
# we then group reviews reviewing the same album (title and artist) then get the average of its rating
df.duplicated(['title'], keep='first').sum()
df.drop_duplicates(['title', 'artist_x'], keep='first').sum()
df.duplicated(['title'], keep='first').sum()
test = df
print(test)
test = test.drop_duplicates(['title','artist_x'], keep='first')
test.duplicated(['title','artist_x'], keep='first').sum()
test.duplicated(['title'], keep='first').sum()
df = test
df['genre'].value_counts()
df[ df['pub_year'] == 2015 ]['genre'].value_counts()
df[ df['pub_year'] == 2012 ]['score'].mean()
df2012 = df[ df['pub_year'] == 2012 ]
df2012['score'].mean()
df[ df['pub_year'] == 2015 ].groupby('genre')['score'].mean()
df[df['pub_year'] == 2015][df['score'] >= 7]['genre'].value_counts()
df[df['pub_year'] == 2015][df['score']==df['score']].max()
df[df['pub_year'] == 2015][df['score'] >= 10]['genre'].value_counts()