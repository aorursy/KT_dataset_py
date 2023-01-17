# import the required libraries

import matplotlib.pyplot as plt

import pandas as pd



# some editing on the display form 

plt.style.use('seaborn-notebook')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/top-10-highest-grossing-films-19752018/blockbusters.csv')

print(df.head())
null = pd.DataFrame(df.isnull().sum() / len(df) * 100).transpose()

print(null)

df.fillna('None', axis=0, inplace=True)




studio = [stu for stu, df in df.groupby('studio')] # to extrat studios' names



# plot studio by rating

plt.barh(studio, df.groupby('studio').mean()['imdb_rating'], label='mean rating (imdb)')



# plot studio by rank

plt.barh(studio, df.groupby('studio').mean()['rank_in_year'], alpha=0.8, label='mean rank')

plt.legend()

plt.title('Rating of Holywood films between (1975-2018)')

plt.xlabel('Rating')

plt.ylabel('Studio name')

plt.show()
# First let's add the year that the film was made to every title for better look (:

df["title"] = df["title"] + ' (' + df["year"].astype(str) + ')'



top_10 = df[['title', 'imdb_rating', 'year', 'length', 'worldwide_gross']].sort_values(by='imdb_rating', ascending=False)

print(top_10.head())
name = [name for name in top_10.iloc[:, 0]] # to extract the names



plt.barh(name[:10], top_10['imdb_rating'][:10], color='lightblue', label='ratings')

plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')

plt.title('Length of the TOP 10 films')

plt.xlabel('length(minutes)')

plt.ylabel('names of the films')
top_10['length'] /= 60 

plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')

plt.title('Length of the TOP 10 films')

plt.xlabel('length(hours)')

plt.ylabel('names of the films')
plt.barh(name[:10], top_10['imdb_rating'][:10], color='lightblue', label='ratings')

plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')



plt.legend()

plt.title('Top 10 films')

plt.xlabel('Rating')

plt.ylabel('Film name')

# preprocessing the feature 

top_10['worldwide_gross'] = top_10['worldwide_gross'].str.replace('$', '')

top_10['worldwide_gross'] = top_10['worldwide_gross'].str.replace(',', '')

# convert it to numeric

top_10['worldwide_gross'] = top_10['worldwide_gross'].astype(float)



plt.barh(name[:10], top_10['worldwide_gross'][:10], color='lightblue', label='worldwide_gross')

plt.title('The worldwide gross of the top 10 films')

plt.xlabel('Worldwide_gross')

plt.ylabel('Top 10 films')

plt.show()


# plot the main genres

main_genre = df.groupby('Main_Genre').mean()

main_genre_names = [gnr for gnr, df in df.groupby('Main_Genre')]

import seaborn as sns

plt.barh(main_genre_names[:10], main_genre['rank_in_year'][:10],alpha=0.8, label='first genre')

######################

# plot the second genres



genre_2 = df.groupby('Genre_2').mean()

genre_names_2 = [gnr for gnr, df in df.groupby('Genre_2')]



plt.barh(genre_names_2[:10], genre_2['rank_in_year'][:10], alpha=0.7, label='second genre')

######################

# plot the third genres



genre_3 = df.groupby('Genre_3').mean()

genre_names_3 = [gnr for gnr, df in df.groupby('Genre_3')]

plt.barh(genre_names_3[:10], genre_3['rank_in_year'][:10], alpha=0.6, label='third genre')



plt.title('Most rated films by genres')

plt.xlabel('Rate')

plt.ylabel('Genres')

plt.legend()

plt.show()