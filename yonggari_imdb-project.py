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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





imd = pd.read_csv('../input/imdb-data/IMDB-Movie-Data.csv')


import warnings

warnings.filterwarnings('ignore')



#RENAME

imdb=imd.rename(columns = {'Revenue (Millions)':'Revenue_Millions', 'Runtime (Minutes)':'Runtime_Minutes'})





#SEE IF NaN VALUES EXISTS

print(imdb.isna().any())



#THIS ONE ADDS UP ALL THE NULL VALUES

print(imdb.isnull().sum())



#FILLS THE NaN WITH ZERO

imdb.fillna(0)



print(imdb.shape)

print(imdb.info())

print(imdb.head())

print(imdb.dtypes)















#GENRE BREAKDOWN

print(imdb["Genre"].value_counts())

seperate_genre='Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western'

for genre in seperate_genre:

    df = imdb['Genre'].str.contains(genre).fillna(False)

    print('The total number of movies with ',genre,'=',len(imdb[df]))















#DIRECTOR BREAKDOWN

imdb.Director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 10 DIRECTORS OF MOVIES')



#AVG DIRECTORS' RATING BY DESCENDING ORDER

directors=imdb.groupby("Director")["Rating"].mean().reset_index()

directors.sort_values("Rating", ascending=False)

print(directors)

















#YEARS BREAKDOWN

#COUNT BY YEAR

print(imdb["Year"].value_counts())

plt.figure(figsize=(10,6))

sns.countplot(data=imdb, x="Year")

plt.ylabel("Number of Movies")









#RUNTIME BREAKDOWN

imdb.Runtime_Minutes.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 10 runtime OF MOVIES')



#BAR GRAPH REPRESENTATION

time=imdb.Runtime_Minutes

sns.distplot(time, bins=20, kde=False, rug=True);









#RATING BREAKDOWN

#BOXPLOT OF RATING

sns.boxplot(imdb.Rating)





#PRINTS THE STATS FOR THE TOP 10 RATED MOVIES

top_rating = imdb.sort_values(['Rating'], ascending=False)

print(top_rating.head(10))



#PRINTS THE STATS FOR THE BOTTOM 10 RATED MOVIES

worst_rating= imdb.query('(Rating > 0) & (Rating < 3.0)')

print(worst_rating.head(10))

print(imdb['Rating'].value_counts())



#PRINTS THE TITLE,DIRECTOR, RATING FOR TOP RATING MOVIE

print(imdb[["Title","Director","Rating"]][imdb["Rating"]==imdb["Rating"].max()])



#COUNT OF HOW MANY MOVIES ARE BETWEEN 7.0 AND 10.0 RATING

high_movie= imdb.query('(Rating > 7.0) & (Rating < 10.0)')

print(len(high_movie))



#COUNT OF HOW MANY MOVIES ARE BETWEEN 3.0 AND 7.0 RATING

med_movie= imdb.query('(Rating > 3.0) & (Rating < 7.0)')

print(len(med_movie))



#CORRELATION BETWEEN CRITICS AND USERS

print(imdb['Rating'].corr(imdb['Metascore']))



#SCATTER PLOT THE CORRELATION BETWEEN CRITICS AND USERS

plt.scatter(imdb.Metascore, imdb.Rating)

plt.show()



#CORRELATION BETWEEN RATING AND REVENUE

print(imdb['Rating'].corr(imdb['Revenue_Millions']))



#SCATTER PLOT THE CORRELATION BETWEEN RATING AND REVENUE

plt.scatter(imdb.Rating, imdb.Revenue_Millions)

plt.show()



#CORRELATION BETWEEN RATING AND RUNTIME

print(imdb['Rating'].corr(imdb['Runtime_Minutes']))



#SCATTER PLOT THE CORRELATION BETWEEN RATING AND RUNTIME

plt.scatter(imdb.Rating, imdb.Runtime_Minutes)

plt.show()



#BAR GRAPH OF MOVIE RATINGS COUNT

movies_ratings = imdb["Rating"]

fig, ax = plt.subplots(figsize=(8,8))

sns.distplot(movies_ratings, kde=True)















#VOTES BREAKDOWN

#MOVIES WITH MORE THAN 1 MILLION VOTES

votes_sorted= imdb.sort_values(['Votes'], ascending=False)



votes= votes_sorted.query('(Votes > 1000000)')

print('number of movies voted more than 1 million :')

print(len(votes))



#THE RATINGS FOR THE 6 MOVIES WITH VOTES MORE THAN 1 MILLION

print(votes_sorted["Rating"].head(6))



#BAR GRAPH OF VOTES BY YEAR

plt.figure(figsize=(10,6))

plt.title("Votes by Years")

sns.barplot(data=imdb, x="Year", y="Votes")

plt.ylabel("Votes")















#REVENUE BREAKDOWN

#TOP 10 REVNUE MOVIES

rev_sorted= imdb.sort_values(['Revenue_Millions'], ascending=False)

print(rev_sorted.head(10))



#MOVIES WITH REVENUE OVER HALF A MILLION

over_half = rev_sorted.query('(Revenue_Millions > 500)')

print(len(over_half))



#PRINTS THE MAX REVENUE MOVIE NAME

print(imdb[imdb["Revenue_Millions"] == imdb["Revenue_Millions"].max()])



#BAR GRAPH OF REVENUE BY YEAR

plt.figure(figsize=(10,6))

plt.title("Revenue by Years")

sns.barplot(data=imdb, x="Year", y="Revenue_Millions")

plt.ylabel("Revenue (Millions)")