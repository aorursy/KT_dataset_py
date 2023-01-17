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





tvs = pd.read_csv('../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
#SEE IF NaN VALUES EXISTS

print(tvs.isna().any())



#THIS ONE ADDS UP ALL THE NULL VALUES

print(tvs.isnull().sum())



#DROPS THE ROWS WITH NaN VALUE

tv2 = tvs.dropna()

#RESETS THE INDEX NUMBER

tv = tv2.reset_index(drop=True)



#GETS RID OF % SIGN FOR ROTTEN TOMATOES

tv['Rotten_tomatoes_%']=tv['Rotten Tomatoes'].str.replace('%', '')



#CHANGES ROTTEN TOMATOES DATA TYPE TO A FLOAT

tv['Rotten_tomatoes_%']=tv['Rotten_tomatoes_%'].astype('float')







print(tv.shape)

print(tv.info())

print(tv.head())

print(tv.dtypes)









#YEAR BREAKDOWN

#PRINTS NUMBER COUNT BY YEAR

print(tv['Year'].value_counts())



#PRINTS NUMBER COUNT BY YEAR FOR EACH PLATFORM

yr_sum = tv.groupby('Year').sum()

print(yr_sum[["Netflix","Hulu","Prime Video","Disney+"]])



#BAR GRAPH OF COUNT BY YEAR

ax = sns.catplot(x='Year',kind='count',data=tv,orient="h",height=30,aspect=1)

ax.fig.suptitle('Number of TV series / movies per year')

ax.fig.autofmt_xdate()











#AGE BREAKDOWN

#PRINTS NUMBER COUNT BY AGE

print(tv['Age'].value_counts())



#PRINTS NUMBER COUNT BY AGE BY PLATFORM

age_platform = tv.groupby('Age').sum()

print(age_platform[["Netflix","Hulu","Prime Video","Disney+"]])



#HORIZONTAL BAR GRAPH OF COUNT BY AGE

sns.catplot(y = 'Age', kind = 'count', palette = 'pastel', edgecolor = '.6', data = tv)











#IMDB RATING BREAKDOWN

#TOP 10 IMDB RATINGS

top = tv.sort_values(by=['IMDb'], ascending = False)

top10 = top.head(10)



#LINE GRAPH OF IMDB RATINGS

sns.kdeplot(data=tv['IMDb'])











#PLATFORM BREAKDOWN

#PRINTS NUMBER COUNT BY PLATFORM

tv_sum= tv.sum()

print(tv_sum["Netflix":"Disney+"])











#SEPARATING EACH PLATFORMS

netflix_movies = tv.loc[tv['Netflix'] == 1]

hulu_movies = tv.loc[tv['Hulu'] == 1]

prime_video_movies = tv.loc[tv['Prime Video'] == 1]

disney_movies = tv.loc[tv['Disney+'] == 1]



#ISOLATING EACH COLUMNS

netflix_movies = netflix_movies.drop(['Hulu', 'Prime Video', 'Disney+', 'Unnamed: 0'], axis = 1)

hulu_movies = hulu_movies.drop(['Netflix', 'Prime Video', 'Disney+', 'Unnamed: 0'], axis = 1)

prime_video_movies = prime_video_movies.drop(['Hulu', 'Netflix', 'Disney+', 'Unnamed: 0'], axis = 1)

disney_movies = disney_movies.drop(['Hulu', 'Prime Video', 'Netflix', 'Unnamed: 0'], axis = 1)







#SET UP FOR ANAYLSIS 

index_netflix = netflix_movies.index

total_netflix_movies = len(index_netflix)



index_hulu = hulu_movies.index

total_hulu_movies = len(index_hulu)



index_prime = prime_video_movies.index

total_prime_movies = len(index_prime)



index_disney = disney_movies.index

total_disney_movies = len(index_disney)











#PIE CHART FOR PLATFORMS BROKEN DOWN

labels = 'Netflix' , 'Hulu', 'Prime Video', 'Disney+'

sizes = [total_netflix_movies,total_hulu_movies,total_prime_movies,total_disney_movies]

explode = (0.1, 0.1, 0.1, 0.1 )



fig1 , ax1 = plt.subplots()



ax1.pie(sizes,

        explode = explode,

        labels = labels,

        autopct = '%1.1f%%',

        shadow = True,

        startangle = 100)



ax1.axis ('equal')

plt.show()











#PLATFORMS WITH MOVIES ABOVE 9+ RATING ON IMDB BY PLATFORMS

rate_mov_net = netflix_movies['IMDb'] > 9

print("Total Movies on Netflix with more than 9+ rating(IMDb) :",rate_mov_net.sum())



rate_mov_dis = disney_movies['IMDb'] > 9

print("Total Movies on Disney+ with more than 9+ rating(IMDb) :",rate_mov_dis.sum())



rate_mov_pvm = prime_video_movies['IMDb'] > 9

print("Total Movies on amazon prime video with more than 9+ rating(IMDb) :",rate_mov_pvm.sum())



rate_mov_hulu = hulu_movies['IMDb'] > 9

print("Total Movies on Hulu with more than 9+ rating(IMDb) :",rate_mov_hulu.sum())