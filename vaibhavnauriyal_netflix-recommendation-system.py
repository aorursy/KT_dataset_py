import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pickle
imdb_df = pd.read_csv('/kaggle/input/netflix-data/IMDb movies.csv')

netflix_df = pd.read_csv('/kaggle/input/netflix-data/netflix_titles.csv')

netflix_df2 = pd.read_csv('/kaggle/input/netflix-data/NetflixViewingHistory.csv')

streaming_platforms_df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
streaming_platforms_df['title']=streaming_platforms_df['Title']

drop=['Unnamed: 0', 'ID','Year', 'Age','Type','Directors','Genres', 'Country', 'Language', 'Runtime','Title','Rotten Tomatoes','IMDb']

streaming_platforms_df.drop(drop, axis=1, inplace=True)
netflix_df2['title']=netflix_df2.Title
drop=['Title','Date']

netflix_df2.drop(drop, axis=1,inplace=True)

netflix_df2 = netflix_df2.drop_duplicates()
imdb_df.columns
drop = ['imdb_title_id','original_title','worlwide_gross_income','metascore','usa_gross_income','budget',

       'writer', 'duration', 'country', 'language', 'director','year', 'date_published']

imdb_df.drop(drop, axis=1, inplace=True)
imdb_df.head()
netflix_df = netflix_df[netflix_df['type']=='Movie']
drop = ['show_id', 'cast', 'country','listed_in','rating','release_year','type','date_added','duration','description']

netflix_df.drop(drop, axis=1, inplace=True)
netflix_df = pd.merge(netflix_df, netflix_df2, how='outer', on='title')

netflix_df = netflix_df.drop_duplicates()

dataset = pd.merge(imdb_df,netflix_df, how='inner',on='title')
dataset.head()
# Calculate all the components based on the weighted averages formula

v=dataset['votes']

R=dataset['avg_vote']

C=dataset['avg_vote'].mean()

m=dataset['votes'].quantile(0.70)
dataset['weighted_average']=((R*v)+ (C*m))/(v+m)
dataset.head()
df_sorted=dataset.sort_values('weighted_average',ascending=False)

df_sorted[['title', 'votes', 'avg_vote', 'weighted_average']].head(20)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

weight_average=df_sorted.sort_values('weighted_average',ascending=False)

plt.figure(figsize=(12,6))

axis1=sns.barplot(x=weight_average['weighted_average'].head(20), y=weight_average['title'].head(20), data=weight_average)

plt.xlim(4, 10)

plt.title('Best Movies on Netflix by average votes(on IMDb)', weight='bold')

plt.xlabel('Weighted Average Score', weight='bold')

plt.ylabel('Movie Title', weight='bold')
dataset['IMDb Score']=dataset['avg_vote']

dataset.drop('avg_vote',axis=1, inplace=True)

dataset.head(1)['description']
def augmentation(df, col1, col2, col3, col4, col5):

    index_col1 = df.columns.get_loc(col1)

    index_col2 = df.columns.get_loc(col2)

    index_col3 = df.columns.get_loc(col3)

    index_col4 = df.columns.get_loc(col4)

    index_col5 = df.columns.get_loc(col5)

    

    for row in range(len(df)):

        count=0

        cast = str(df.iat[row, index_col2])

        main_cast = ""

        for i in range(len(cast)):

            if cast[i]!=',':

                if count!=3:

                    main_cast = main_cast+cast[i]

                else:

                    break

            else:

                count=count+1

        df.iat[row,index_col3] = str(str(df.iat[row,index_col1])+str(main_cast)+str(df.iat[row,index_col4])+str(df.iat[row, index_col5]))

        

dataset["Information"]=""



augmentation(dataset,'description','actors','Information','genre','director')
def case_conversion(df, col1, col2):

    index_col1 = df.columns.get_loc(col1)

    index_col2 = df.columns.get_loc(col2)

    

    for rows in range(len(df)):

        df.iat[rows, index_col2] = df.iat[rows, index_col1].lower()

        

dataset['title_lower'] = ""

case_conversion(dataset, "title", "title_lower")
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 6),

            stop_words = 'english')



# Filling NaNs with empty string

dataset['Information'] = dataset['Information'].fillna('')

dataset['description'] = dataset['description'].fillna('None')
dataset.to_csv('movie_dataset.csv', header=True, index=False)
# Fitting the TF-IDF on the 'Information' text

tfv_matrix = tfv.fit_transform(dataset['Information'])
from sklearn.metrics.pairwise import sigmoid_kernel



# Compute the sigmoid kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles

indices = pd.Series(dataset.index, index=dataset['title_lower']).drop_duplicates()


def recommendations(title, sig=sig):

    # Get the index corresponding to original_title

    title = title.lower()

    idx = indices[title]



    # Get the pairwsie similarity scores 

    sig_scores = list(enumerate(sig[idx]))



    # Sort the movies 

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)



    # Scores of the 10 most similar movies

    sig_scores = sig_scores[1:11]



    # Movie indices

    movie_indices = [i[0] for i in sig_scores]



    # Top 10 most similar movies

    return dataset.iloc[movie_indices]

    
df = recommendations("the green mile")

data = df[['title','genre','description','IMDb Score','actors']].head(10)

data