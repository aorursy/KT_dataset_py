import pandas as pd
netflix_raw_df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
import numpy as np
netflix_raw_df
netflix_raw_df.columns
netflix_raw_df.shape
netflix_raw_df.info()
netflix_raw_df.describe()
selected_columns = [

    'show_id',

    'type',

    'title',

    'director',

    'country',

    'date_added',

    'release_year',

    'rating',

    'listed_in',

    'cast'

    

]
netflix_df = netflix_raw_df[selected_columns].copy()
netflix_df
netflix_df.listed_in.unique()
netflix_df.where(~(netflix_df.listed_in.str.contains(',', na=False)), np.nan, inplace=True)
netflix_df.where(~(netflix_df.listed_in.str.contains('&', na=False)), np.nan, inplace=True)
netflix_df.listed_in.unique()
# Select a project name

project='Netflix_shows_survey'
# Install the Jovian library

!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project)
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

shows_fname=netflix_df.release_year.value_counts()
shows_fname
plt.figure(figsize=(15,8))

plt.xticks(rotation=75)

plt.title('SHOWS PER YEAR')

plt.xlabel('Year')

plt.ylabel('No. of shows')

sns.barplot(shows_fname.index, shows_fname);
shows_per_category_counts = netflix_df.listed_in.value_counts()
plt.figure(figsize=(15,8))

sns.countplot(y=netflix_df.listed_in)

plt.xticks(rotation=75);

plt.title('shows_per_genre')

plt.ylabel(None);
types_fname=netflix_df.type.value_counts()
types_fname
sns.scatterplot('type', 'listed_in', data=netflix_df)

plt.xlabel("type")

plt.ylabel("category");
director_f=netflix_df.director.value_counts().head(10)
director_f
plt.figure(figsize=(15,10))

plt.xticks(rotation=75)

plt.title('director vs no. of shows')

plt.xlabel('director')

plt.ylabel('No. of shows')

sns.barplot(director_f.index, director_f);
Movie_df = netflix_df[netflix_df.type == 'Movie']

TV_Show_df = netflix_df[netflix_df.type == 'TV Show']
plt.title('type vs year')



plt.hist([Movie_df.release_year, TV_Show_df.release_year], 

         

         stacked=True);



plt.legend(['Movie', 'TV_Show']);
import jovian
jovian.commit(project=project)
netflix_df.listed_in.value_counts()
doc_pr=299/6234*100
doc_pr
nt=netflix_df.release_year.value_counts()
plt.xlabel('year')

plt.ylabel('count')

plt.plot(netflix_df.release_year.value_counts());

total_produced_df=Movie_df.count()+TV_Show_df.count()
movies_produced_percentages = (Movie_df.title.count() * 100/ total_produced_df.type)
movies_produced_percentages
netflix_df.rating.unique()
rating_df = netflix_df[netflix_df.rating == 'TV-Y7-FV']
rating_df.title.count()
order =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']

plt.figure(figsize=(15,7))

g = sns.countplot(netflix_df.rating, hue=netflix_df.type, order=order, palette="pastel");

plt.title("Ratings for Movies & TV Shows")

plt.xlabel("Rating")

plt.ylabel("Total Count")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(19, 5))

g1 = sns.countplot(Movie_df.rating, order=order,palette="Set2", ax=ax[0]);

g1.set_title("Ratings for Movies")

g1.set_xlabel("Rating")

g1.set_ylabel("Total Count")

g2 = sns.countplot(TV_Show_df.rating, order=order,palette="Set2", ax=ax[1]);

g2.set(yticks=np.arange(0,1600,200))

g2.set_title("Ratings for TV Shows")

g2.set_xlabel("Rating")

g2.set_ylabel("Total Count")

fig.show()
import jovian
jovian.commit(project=project)
import jovian
jovian.commit(project=project)