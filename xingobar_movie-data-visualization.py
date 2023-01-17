# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/movie_metadata.csv')
df.head()
df['diff_gross'] = df['gross'] - df['budget']

df_copy = df.copy().dropna()

director_budge = df_copy.groupby(df_copy['director_name'])['diff_gross'].sum()

direcotr_budge_indx = director_budge.sort_values(ascending=False)[:20].index

director_budge_pivot = pd.pivot_table(data = df_copy[df_copy['director_name'].isin(direcotr_budge_indx)],

                                      index=['title_year'],

                                      columns=['director_name'],

                                      values=['diff_gross'],

                                      aggfunc='sum')





fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(director_budge_pivot['diff_gross'],vmin=0,annot=False,linewidth=.5,ax=ax,cmap='PuBu')

plt.title('Director vs Year and diff_gross')

plt.ylabel('Year')
director_critic_counts = df_copy.groupby(df_copy['director_name'])['num_critic_for_reviews'].sum()

director_critic_indx = director_critic_counts.sort_values(ascending=False)[:20].index

director_critic_values = director_critic_counts.sort_values(ascending=False)[:20].values



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = director_critic_indx,

            y = director_critic_values,

            color='#90caf9',

            ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Director vs critic')

plt.ylabel('Counter')

plt.xlabel('Director')

del fig,ax,ticks
country_gross = df_copy.groupby(df['country'])['diff_gross'].sum().sort_values(ascending=False)

country_gross_indx = country_gross[:20].index



country_pivot = pd.pivot_table(data = df_copy[df_copy['country'].isin(country_gross_indx)],

                               index=['title_year'],

                               columns=['country'],

                               values=['diff_gross'],

                               aggfunc='sum')

fig,ax = plt.subplots(figsize=(8,10))

sns.heatmap(country_pivot['diff_gross'],vmin=0,linewidth=.5,annot=False,cmap='PuBu',ax=ax)

plt.title('Country\'s diff_gross vs year')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

del fig,ax,ticks
df_copy['critic_ratio'] = df_copy['num_critic_for_reviews'] / df_copy['num_user_for_reviews']

df_copy.head()
director_critic_ratio = df_copy.groupby(df_copy['director_name'])['critic_ratio'].mean()

director_critic_idx =director_critic_ratio.sort_values(ascending=False)[:20].index

director_critic_val =director_critic_ratio.sort_values(ascending=False)[:20].values



director_critic_pivot = pd.pivot_table(data=df_copy[df_copy['director_name'].isin(director_critic_idx)],

                                       index=['title_year'],

                                       columns=['director_name'],

                                       values=['critic_ratio'],

                                       aggfunc='mean')

fig,ax = plt.subplots(figsize=(8,10))

sns.heatmap(director_critic_pivot['critic_ratio'],vmin=0,annot=False,linewidth=.5,ax=ax)

plt.title('Top 20 critic ratio')

plt.ylabel('Year')

plt.xlabel('Director')
movie_genres = df_copy['genres'].map(lambda x:x.split('|'))

genres = []

for genre in movie_genres:

    if(len(genre) >= 2):

        for i in genre:

            if i not in genres:

                genres.append(i)

    elif genre not in genres:

        if isinstance(genre,list):

            genres.append(genre[0])

genres = set(genres)











def process_genre(genres):

    genre_list = []

    for genre in genres.split('|'):

        genre_list.append(genre)

    return genre_list

df_copy['genre_list'] = df_copy['genres'].map(process_genre)

df_copy.head()









df_copy = df_copy.reset_index(drop=True)

total_genre_list = []

for idx in range(len(df_copy)):

    for genre in df_copy['genre_list'][idx]:

        total_genre_list.append(genre)

genre_counter =Counter(total_genre_list)

genre_counter_indx = np.asarray(list(genre_counter.keys()))

genre_counter_val = np.asarray(list(genre_counter.values()))



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = genre_counter_indx, y = genre_counter_val,color='#90caf9',ax=ax)

plt.title('Total Genre')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

del fig,ax,ticks
