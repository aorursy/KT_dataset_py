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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')



print(movies.describe())

movies.info()

print(movies.columns)

print(movies.dtypes)

print(movies.head())

print(movies.status.head(15))

print(movies.corr())

sns.heatmap(movies.corr(), annot = True, linewidths = .5, fmt = '.1f')

plt.show()







#BUDGET BREAKDOWN

print(movies.budget.mean())

print(movies.budget.std())

print(movies.budget.median())

print(movies.budget.max())

print(movies.budget.min())

print(movies.budget.sort_values())  #sorts the budget from lowest to highest

#plt.plot(movies.title, movies.budget, color = 'blue') #graph of title, by budget









#RUNTIME BREAKDOWN

print(movies.runtime.mean())

print(movies.runtime.sort_values()) #sorts from lowest runtime to highest

print(movies.runtime.describe())

print(movies.runtime.median())

movies.runtime.plot.hist()









#GENRES BREAKDOWN

print(movies.genres.nunique())

print(movies.genres.mode())









#ORIGINAL LANGUAGE BREAKDOWN

print(movies['original_language'].value_counts())

print(movies.original_language.mode())













#RELEASE DATE BREAKDOWN

print(movies.release_date.sort_values())













#POPULARITY BREAKDOWN

print(movies.popularity.describe())

print(np.percentile(movies.popularity,99)) #find 99th perentile

pop_max = movies.popularity.max()

print(movies.loc[movies['popularity'] == pop_max, 'original_title']) #prints out most popular movie title











#STATUS OF MOVIE BREAKDOWN

print(movies.status.unique())

print(movies.status.count())

print(movies['status'].value_counts())  #prints how many for each status

print(movies.loc[movies['status'] == 'Post Production', 'original_title']) #prints the names of post-production movies

print(movies.loc[movies['status'] == 'Rumored', 'original_title'])













#Plots a graph of average votes

vt10 = movies.query('vote_count >= 10')

g1 = sns.distplot(vt10.vote_average, norm_hist = False, kde=False)

g1.set(label = "Vote Average", ylabel = 'Frequency')

g1.set_title('Average Votes for 10+ Votes')



#VOTE AVERAGE BREAKDOWN

print(movies.vote_average.describe())

print(np.percentile(movies.vote_average, 90)) #find 90th percentile

v_max = movies.vote_average.max()

print(movies.loc[movies['vote_average'] == v_max, 'original_title']) 

v_max = vt10.vote_average.max()

print(movies.loc[movies['vote_average'] == v_max, 'original_title']) 







a = sns.distplot(vt10.vote_average, hist_kws = {'cumulative':True}, kde_kws = {'cumulative':True})

a.set(label='Average Votes', ylabel='Cumulative Ratio')

a.set_title('Average Rating of TMDB')



b = sns.distplot(vt10.query('budget > 0').budget)

b.set(label = 'Budget', ylabel = 'Density')

b.set_title ('Budget on movies - TMDB')



#Plots a box plot of average rating

a = sns.boxplot(movies.vote_average)

a.set(label = 'Average Rating')

a.set_title('Average Rating Distribution')



c = sns.distplot(vt10.query('runtime > 0').runtime.fillna(vt10.runtime.mean()))

c.set(label = 'Runtime', ylabel = 'Density')

c.set_title ('Runtime on movies - TMDB')







#PLOT REVENUE BY vote_average

max_rev = movies.revenue.max()

print(movies.loc[movies['revenue'] == max_rev, 'original_title']) #prints title of ma revenue movie

print(movies.revenue.mean())

plt.scatter([movies.vote_average], [movies.revenue]) 

plt.ylabel('Revenue')

plt.xlabel('Voted Score')



movies.revenue.plot(kind='line', color='r', label='revenue', linewidth=.7, alpha=.5, grid=True, linestyle='-' )

movies.budget.plot(color='g', label='budget', linewidth=.7, alpha=.8, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('Film ID')

plt.ylabel('Money')

plt.title('Line Plot')

plt.show()









movies.plot(kind='scatter', x ='vote_average', y='budget', alpha=.5, color='r')

plt.xlabel('vote_average')

plt.ylabel('budget')

plt.title('Scatter Plot')

plt.show()







movies.budget.plot(kind='hist', bins = 20, figsize = (10,10))

plt.show()