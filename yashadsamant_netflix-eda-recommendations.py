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

from tqdm import tqdm

import seaborn as sns

from scipy import spatial

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# load data

# path = "C:/Users/gunjan/Google Drive/Kaggle/Netflix/data/"

path = '/kaggle/input/netflix-shows/netflix_titles.csv'
data_df = pd.read_csv(path)

data_df.head()
year_x = data_df['release_year'].values

fig = sns.distplot(year_x,kde=False)

plt.xlabel("Year of Release")

plt.ylabel("Frequency")

plt.title("Distribution of Netflix Shows over Release Year")

plt.show(fig)
pd.value_counts(data_df['type']).plot(kind="bar")

plt.xlabel("Type of Show")

plt.ylabel("Frequency")

plt.title("Distribution of Netflix Shows over Type of Show")

plt.show()
country=data_df.groupby('country').count()

country.sort_values(by='show_id', inplace=True, ascending=False)

country_top=country.head(10)



country_top['show_id'].plot(kind='barh', figsize=(11,15))

plt.title("Distribution of Netflix Shows over Country")

plt.xlabel('Frequency')

plt.ylabel('Country')

plt.show()
genre=data_df.groupby('listed_in').count()

genre.sort_values(by='show_id', inplace=True, ascending=False)

genre_top=genre.head(10)



genre_top['show_id'].plot(kind='barh', figsize=(11,15))

plt.title("Distribution of Netflix Shows over Genre")

plt.xlabel('Frequency')

plt.ylabel('Genre')

plt.show()
def clean_country(country_list):

    for i, country in enumerate(country_list):

        country_list[i] = country.strip()

    return set(country_list)



def find_country_score(movie_1, movie_2):

    try:

        country_m1 = movie_1['country'].split(',')

        country_m2 = movie_2['country'].split(',')

        country_m1 = clean_country(country_m1)

        country_m2 = clean_country(country_m2)

        union = len(country_m1.union(country_m2))

        inter = len(country_m1.intersection(country_m2)) 

        return inter/union

    except Exception as e:

        return 0.0

    

# test country similarity



country_sim = []

num = 50

test_data = data_df.head(num)

for i, row1 in test_data.iterrows():

    row_sim = []

    for j, row2 in test_data.iterrows():

        row_sim.append(find_country_score(row1, row2))

    country_sim.append(row_sim)
df_cm = pd.DataFrame(country_sim, range(num), range(num))

plt.figure(figsize=(10,10))

sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 5}) # font size



plt.show()
ratings = data_df['rating'].unique()

replace_rating = {}

for rating in ratings:

    if rating == 'TV-PG' or rating == 'PG' or rating == 'PG-13' or rating == 'TV-14':

        replace_rating[rating] = 'PG'

    

    elif rating == 'TV-MA' or rating == 'NC-17' or rating == 'R':

        replace_rating[rating] = 'R'

    

    elif rating == 'NR' or rating == 'UR' or rating == 'TV-G' or rating == 'G':

        replace_rating[rating] = 'U'

    

    elif rating == 'TV-Y7-FV' or rating == 'TV-Y7' or rating == 'TV-Y':

        replace_rating[rating] = 'Y'

    

    else:

        replace_rating[rating] = 'NAN'

replace_rating



data_df['rating'] = data_df['rating'].map(replace_rating)

data_df['rating']
def find_rating_score(movie_1, movie_2):

    rating_1 = movie_1['rating']

    rating_2 = movie_2['rating']

    if rating_1 == 'U':

        recom = 1.0

    

    elif rating_1 == 'Y':

        if rating_2 == 'R':

            recom = 0.0

        elif rating_2 == 'PG':

            recom = 0.25

        elif rating_2 == 'U':

            recom = 1.0

        else:

            recom = 0.0



    elif rating_1 == 'PG':

        if rating_2 == 'R':

            recom = 0.0

        elif rating_2 == 'Y':

            recom = 0.75

        elif rating_2 == 'U':

            recom = 1.0

        else:

            recom = 0.0

            

    elif rating_1 == 'R':

        if rating_2 == 'Y':

            recom = 0.25

        elif rating_2 == 'PG':

            recom = 0.5

        elif rating_2 == 'U':

            recom = 1.0

        else:

            recom = 0.0

    

    else:

        recom = 0.0

    

    return recom 



rating_sim = []

num = 50

test_data = data_df.head(num)

for i, row1 in test_data.iterrows():

    row_sim = []

    for j, row2 in test_data.iterrows():

        row_sim.append(find_rating_score(row1, row2))

    rating_sim.append(row_sim)
df_cm = pd.DataFrame(rating_sim, range(num), range(num))

plt.figure(figsize=(10,10))

sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 5}) # font size



plt.show()
genre = data_df['listed_in']

genre = genre.values

list_genre = []

for g in genre:

    for i in g.split(','):

        list_genre.append(i.strip())



list(set(list_genre))
def clean_genre(genre_list):

    for i, genre in enumerate(genre_list):

        genre_list[i] = genre.strip()

    return set(genre_list) 



def find_genre_score(movie_1, movie_2):

    try:

        genre_m1 = movie_1['listed_in'].split(',')

        genre_m2 = movie_2['listed_in'].split(',')

        genre_m1 = clean_genre(genre_m1)

        genre_m2 = clean_genre(genre_m2)

        union = len(genre_m1.union(genre_m2))

        inter = len(genre_m1.intersection(genre_m2)) 

        return inter/union

    except Exception as e:

        return 0.0

    

# test country similarity



genre_sim = []

num = 50

test_data = data_df.head(num)

for i, row1 in test_data.iterrows():

    row_sim = []

    for j, row2 in test_data.iterrows():

        row_sim.append(find_genre_score(row1, row2))

    genre_sim.append(row_sim)
df_cm = pd.DataFrame(genre_sim, range(num), range(num))

plt.figure(figsize=(10,10))

sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 5}) # font size



plt.show()
# "stopwords" are the words that appear very frequently in a language and have very low importance in determining the context of the sentence. It is a good practice to remove these stopwords before we start processing the description for TF-IDF.

#  define all the stop words

stopwords = ['i',

'me',

'my',

'myself',

'we',

'our',

'ours',

'ourselves',

'you',

'your',

'yours',

'yourself',

'yourselves',

'he',

'him',

'his',

'himself',

'she',

'her',

'hers',

'herself',

'it',

'its',

'itself',

'they',

'them',

'their',

'theirs',

'themselves',

'what',

'which',

'who',

'whom',

'this',

'that',

'these',

'those',

'am',

'is',

'are',

'was',

'were',

'be',

'been',

'being',

'have',

'has',

'had',

'having',

'do',

'does',

'did',

'doing',

'a',

'an',

'the',

'and',

'but',

'if',

'or',

'because',

'as',

'until',

'while',

'of',

'at',

'by',

'for',

'with',

'about',

'against',

'between',

'into',

'through',

'during',

'before',

'after',

'above',

'below',

'to',

'from',

'up',

'down',

'in',

'out',

'on',

'off',

'over',

'under',

'again',

'further',

'then',

'once',

'here',

'there',

'when',

'where',

'why',

'how',

'all',

'any',

'both',

'each',

'few',

'more',

'most',

'other',

'some',

'such',

'no',

'nor',

'not',

'only',

'own',

'same',

'so',

'than',

'too',

'very',

's',

't',

'can',

'will',

'just',

'don',

'should',

'now']
from sklearn.feature_extraction.text import TfidfVectorizer



# extract all the descriptions

descriptions = data_df['description'].values

des = list(descriptions)



# cerate the vectorizer with given stopwords

vectorizer = TfidfVectorizer(stop_words = stopwords)

X = vectorizer.fit_transform(des)

XX = X.todense()

print(XX.shape)

## XX is the matrix which stores TFIDF scores for each description
# function to calculate similarity score using TFIDF on description



def find_description_score(movie_1, movie_2):

    

    '''

    INPUT: movie_1 : row2 for features of the show1 given by the User & 

           movie_2 : row2 for features of the show2 we are comparing this show with

    OUTPUT: Similarity score between the descriptions of show1 and show2

    '''

    

    # extract row numbers

    index_1 = data_df.index[data_df['title'] == movie_1['title']]

    index_2 = data_df.index[data_df['title'] == movie_2['title']]

    

    a = np.array(XX[index_1])

    b = np.array(XX[index_2])

    similarity_score = 1-spatial.distance.cosine(a[0], b[0])

    

    return similarity_score

movie_1 = data_df.iloc[17]

print(movie_1)



def find_recommendations(data_df, movie_1):

    rec_score = []

    for index, movie_2 in tqdm(data_df.iterrows()):

        rec_country = find_country_score(movie_1, movie_2)

        rec_genre = find_genre_score(movie_1, movie_2)

        rec_rating = find_rating_score(movie_1, movie_2)

        rec_description = find_description_score(movie_1, movie_2)



        score = 0.2*rec_genre + 0.2*rec_rating + 0.2*rec_country + 0.4*rec_description

        rec_score.append(score)

    

    data_df['score'] = pd.DataFrame(rec_score)   

    return data_df



data_df = find_recommendations(data_df, movie_1)

data_df.sort_values(by=['score'], ascending=False).head(10)