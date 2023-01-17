import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display # Allows the use of display() for DataFrames

from time import time

import matplotlib.pyplot as plt

import seaborn as sns # Plotting library

from scipy import stats

from collections import Counter

import json

from sklearn.metrics import mean_squared_error

from math import sqrt

import lightgbm as lgb



%matplotlib inline



import os

print(os.listdir("../input"))



# Always show all columns

pd.set_option('display.max_columns', 999)
# Open the train and test data into a dataframe.

data_train_raw = pd.read_csv("../input/train.csv")

data_test_raw = pd.read_csv("../input/test.csv")



# Display the data to see what it looks like

display(data_train_raw.head(n=3))



display(data_test_raw.head(n=3))
display(data_train_raw.describe())
# Revenues are skewed

fig, ax = plt.subplots(figsize=(10,5))

fig.suptitle('Revenue Distribution', fontsize=15)

sns.distplot(data_train_raw['revenue'], bins=50, kde=False)

ax.grid()
# Check the distribution of some of the continous features

fig, ax = plt.subplots(ncols=3, figsize=(20,5))

fig.suptitle('Distribution of continuous features', fontsize=15)



features_to_explore = ['budget', 'popularity', 'runtime']



for col, feature in enumerate(features_to_explore):

    myplot = sns.distplot(data_train_raw[feature], bins=50, kde=False, ax=ax[col])

    myplot.grid()
# Check correlation between revenue and some of the continous features

fig, ax = plt.subplots(ncols=3, figsize=(20,5))

fig.suptitle('Correlation between some continuous features\n and target feature `revenue`', fontsize=15)



for col, feature in enumerate(features_to_explore):

    myplot = sns.regplot(x='revenue', y=feature, data=data_train_raw, ax=ax[col])
## Train data

data_train_preprocessed = data_train_raw.copy()

## Test data

data_test_preprocessed = data_test_raw.copy()
# Swap NaN values with the mean

def runtime_pre_process(df):

    df['runtime'] = df['runtime'].fillna(0)

    

    # Swap runtime with 0 values for the average

    runtime_mean = df['runtime'].mean()

    

    df['runtime'] = df['runtime'].replace(0, runtime_mean)

    

    return df
data_train_preprocessed = runtime_pre_process(data_train_preprocessed)

data_test_preprocessed = runtime_pre_process(data_test_preprocessed)



# Before

print(data_train_raw['runtime'].loc[data_train_raw['runtime'] == 0].count())

# After

print(data_train_preprocessed['runtime'].loc[data_train_preprocessed['runtime'] == 0].count())
print(data_test_preprocessed['release_date'].value_counts()[:1])

display(data_test_preprocessed['release_date'].isna().value_counts())

# Fill any NaN values with  the most common Value

data_test_preprocessed.loc[data_test_preprocessed['release_date'].isnull() == True, 'release_date'] = '01/01/98'
def release_year_pre_process (df):

    # Fill any NaN values

    df.loc[df['release_date'].isnull() == True, 'release_date'] = '01/01/98'

    

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_month'] = df['release_date'].apply(lambda d: d.month)

    df['release_day'] = df['release_date'].apply(lambda d: d.day)

    df['release_weekday'] = df['release_date'].apply(lambda d: d.weekday())



    # For some reason some dates were put into the future

    df['release_year'] = df['release_date'].apply(lambda d: d.year if d.year < 2018 else d.year -100)

    

    return df
data_train_preprocessed = release_year_pre_process(data_train_preprocessed)

data_test_preprocessed = release_year_pre_process(data_test_preprocessed)



display(data_train_preprocessed.head(n=1))
def homepage_pre_process(df):

    # Replace NaN values with 0

    df['homepage'] = df['homepage'].fillna(0)

    

    # Replace rows with websites with 1

    df.loc[df['homepage'] != 0, 'homepage'] = 1

    

    return df
data_train_preprocessed = homepage_pre_process(data_train_preprocessed)

data_test_preprocessed = homepage_pre_process(data_test_preprocessed)



display(data_train_preprocessed['homepage'].head())
def poster_pre_process(df):

    # Replace NaN values with 0

    df['poster_path'] = df['poster_path'].fillna(0)

    

    # Replace rows with websites with 1

    df.loc[df['poster_path'] != 0, 'poster_path'] = 1

    

    return df
data_train_preprocessed = poster_pre_process(data_train_preprocessed)

data_test_preprocessed = poster_pre_process(data_test_preprocessed)



display(data_train_preprocessed['poster_path'].head())
# Count how many films have budget 0

print(data_train_preprocessed['budget'].loc[data_train_preprocessed['budget'] == 0].count())



print(data_test_preprocessed['budget'].loc[data_test_preprocessed['budget'] == 0].count())
fig, axs = plt.subplots(figsize=(8,8))

fig.suptitle('Correlation Matrix', fontsize=16)

sns.heatmap(data_train_preprocessed.corr(), annot=True, ax=axs)
# Check for correlations between budget and other features

numerical_features = ['budget', 'popularity', 'poster_path', 'homepage', 'release_year', 'release_month', 'release_day', 'runtime']



correlation_budget = []

for column in data_train_preprocessed[numerical_features].loc[data_train_preprocessed['budget'] != 0]:

    pearson_corr, _ = stats.pearsonr(data_train_preprocessed['budget'], data_train_preprocessed[column])

    correlation_budget.append(('budget', column, pearson_corr))

    

display(pd.DataFrame(data=correlation_budget, columns=['Start Feature', 'Target', 'Correlation value']))
# We will input the budget with ceros with the year's average

# Also we will add a column that contains that year's budget average for all the movies release in that year

def imputing_budget(df):

    df['budget'] = df['budget'].fillna(0)

    year_mean = df.groupby(['release_year']).mean()['budget']

    df['year_mean_budget'] = 0

    

    for index, row in df.iterrows():

        year_of_release = row['release_year']

        

        if row['budget'] == 0:

            df.at[index,'budget'] = year_mean[year_of_release]

        

        # Average budget that year

        df.at[index,'year_mean_budget'] = year_mean[year_of_release]

            

    return df
# Update our data

data_train_preprocessed = imputing_budget(data_train_preprocessed)

data_test_preprocessed = imputing_budget(data_test_preprocessed)



# Check how many values have 0 now.

print(data_train_preprocessed['budget'].loc[data_train_preprocessed['budget'] == 0].count())

print(data_test_preprocessed['budget'].loc[data_test_preprocessed['budget'] == 0].count())
# Count how many ´belongs to collection´ have NaN values

print(data_train_preprocessed['belongs_to_collection'].head())

print('\n')

print(data_train_preprocessed['belongs_to_collection'].isna().value_counts())
def collection_pre_process(df):

    # Change NaN values with ´0´.

    df['belongs_to_collection'] = df['belongs_to_collection'].fillna('0')

    

    # Replace rows that belong to collection with 1

    df.loc[df['belongs_to_collection'] != 0, 'belongs_to_collection'] = 1

    

    return df
data_train_preprocessed = collection_pre_process(data_train_preprocessed)

data_test_preprocessed = collection_pre_process(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['belongs_to_collection'].isna().value_counts())

display(data_train_preprocessed.head(n=1))
# Change NaN values with the most common value

print(data_train_preprocessed['genres'].isna().value_counts())

print('\n')

print(data_train_preprocessed['genres'].value_counts()[:5])
def genres_pre_processing(df):

    # Most common value is Drama, so I'll fill empty values with [{'id': 18, 'name': 'Drama'}]

    df['genres'] = df['genres'].fillna("[{'id': 18, 'name': 'Drama'}]")

    

    return df
data_train_preprocessed = genres_pre_processing(data_train_preprocessed)

data_test_preprocessed = genres_pre_processing(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['genres'].isna().value_counts())
# No missing values in train

print(data_train_preprocessed['original_title'].isna().value_counts())



# No missing values in test

print(data_test_preprocessed['original_title'].isna().value_counts())
# Fill ´none´ titles with the same value as ´original_title´

def imput_title(df):

    df['title'] = df['title'].fillna("none")



    for index, row in df.iterrows():

        if row['title'] == "none":

            df.at[index,'title'] = df.loc[index]['original_title']

    return df        
data_train_preprocessed = imput_title(data_train_preprocessed)

data_test_preprocessed = imput_title(data_test_preprocessed)



# Check Nan for train

print(data_train_preprocessed['original_title'].isna().value_counts())
# No missing values in train

print(data_train_preprocessed['original_language'].isna().value_counts())



# No missing values in test

print(data_test_preprocessed['original_language'].isna().value_counts())
def prod_comp_pre_processing(df):

    # Replace NaN values with 'none'

    df['production_companies'] = df['production_companies'].fillna("none")

    

    return df
data_train_preprocessed = prod_comp_pre_processing(data_train_preprocessed)

data_test_preprocessed = prod_comp_pre_processing(data_test_preprocessed)



# Check for NaNs

print(data_train_preprocessed['production_companies'].isna().value_counts())

print(data_test_preprocessed['production_companies'].isna().value_counts())
def spoken_lang_pre_processing(df):

    # Replace NaN values with the most common [{'iso_639_1': 'en', 'name': 'English'}]

    df['spoken_languages'] = df['spoken_languages'].fillna("[{'iso_639_1': 'en', 'name': 'English'}]")

    

    return df
data_train_preprocessed = spoken_lang_pre_processing(data_train_preprocessed)

data_test_preprocessed = spoken_lang_pre_processing(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['spoken_languages'].isna().value_counts())

print(data_test_preprocessed['spoken_languages'].isna().value_counts())
def keywords_pre_processing(df):

    # Replace NaN values with 'none'

    df['Keywords'] = df['Keywords'].fillna("none")

    

    return df
data_train_preprocessed = keywords_pre_processing(data_train_preprocessed)

data_test_preprocessed = keywords_pre_processing(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['Keywords'].isna().value_counts())

print(data_test_preprocessed['Keywords'].isna().value_counts())
def cast_pre_processing(df):

    # Replace NaN values with 'none'

    df['cast'] = df['cast'].fillna("none")

    

    return df
data_train_preprocessed = cast_pre_processing(data_train_preprocessed)

data_test_preprocessed = cast_pre_processing(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['cast'].isna().value_counts())

print(data_test_preprocessed['cast'].isna().value_counts())
def crew_pre_processing(df):

    # Replace NaN values with 'none'

    df['crew'] = df['crew'].fillna("none")

    

    return df
data_train_preprocessed = crew_pre_processing(data_train_preprocessed)

data_test_preprocessed = crew_pre_processing(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['crew'].isna().value_counts())

print(data_test_preprocessed['crew'].isna().value_counts())
def overview_pre_process(df):

    # Replace NaN values with ""

    df['overview'] = df['overview'].fillna("")

    

    return df
data_train_preprocessed = overview_pre_process(data_train_preprocessed)

data_test_preprocessed = overview_pre_process(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['overview'].isna().value_counts())

print(data_test_preprocessed['overview'].isna().value_counts())
def tagline_pre_process(df):

    # Replace NaN values with ''

    df['tagline'] = df['tagline'].fillna("")

    

    return df
data_train_preprocessed = tagline_pre_process(data_train_preprocessed)

data_test_preprocessed = tagline_pre_process(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['tagline'].isna().value_counts())

print(data_test_preprocessed['tagline'].isna().value_counts())
data_train_preprocessed['production_countries'].value_counts()[:5]
def prod_countries_pre_process(df):

    # Replace NaN values with most common

    df['production_countries'] = df['production_countries'].fillna("[{'iso_3166_1': 'US', 'name': 'United States of America'}]")

    

    return df
data_train_preprocessed = prod_countries_pre_process(data_train_preprocessed)

data_test_preprocessed = prod_countries_pre_process(data_test_preprocessed)



# Check for NaN values

print(data_train_preprocessed['production_countries'].isna().value_counts())

print(data_test_preprocessed['production_countries'].isna().value_counts())
print("NaN values in train['status'] : {}".format(data_train_preprocessed['status'].isna().any()))

print("NaN values in test['status'] : {}".format(data_test_preprocessed['status'].isna().any()))

print('\nValue Counts train:')

print(data_train_preprocessed['status'].value_counts())
def status_pre_process(df):

    df['status'] = df['status'].fillna("Released")

    

    return df
data_train_preprocessed = status_pre_process(data_train_preprocessed)

data_test_preprocessed = status_pre_process(data_test_preprocessed)



display(data_test_preprocessed['status'].value_counts())
import ast



# Convert panda list string to actual list

string_lists = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew', 'Keywords']



def string2_list(df):

    for string_list in string_lists:

        df[string_list] = df[string_list].apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

        

    return df

        

data_train_preprocessed = string2_list(data_train_preprocessed)

data_test_preprocessed = string2_list(data_test_preprocessed)
display(data_train_preprocessed.isna().any())

display(data_test_preprocessed.isna().any())
# Drop some columns that we won't use

data_train_cleaned = data_train_preprocessed.copy()

data_test_cleaned = data_test_preprocessed.copy()



data_train_cleaned = data_train_cleaned.drop(['imdb_id'], axis=1)

data_test_cleaned = data_test_cleaned.drop(['imdb_id'], axis=1)
# Flattens a a list of lists of dicts into a simple list of dicts

def flatten_data_column(mylist):

    flattened_list = []

    for elements in mylist:

        for element in elements:

            flattened_list.append(element)

    return flattened_list

# Groups by repeated value and counts the ocurrences

def create_counter(mylist, key='name'):

    return Counter([i[key] for i in mylist]).most_common()
# Lets get the list of genres

genres_list = flatten_data_column(data_train_cleaned['genres'])



genres_list_test = flatten_data_column(data_test_cleaned['genres'])
# Create a counter of the most popular genres

genres_list_counter = create_counter(genres_list)

genres_list_counter_test = create_counter(genres_list_test)
def genres_FE(df, genres_counter):

    df['num_genres'] = df['genres'].apply(lambda x: len(x) if x != {} else 0)

    df['all_genres'] = df['genres'].apply(lambda x: [i['name'] for i in x])

    

    for genre, count in genres_counter:

        df['genre_' + genre] = df['all_genres'].apply(lambda g_list: 1 if genre in g_list else 0)

    

    df['all_genres'] = df['all_genres'].apply(lambda x: " ".join(x))

    df = df.drop(['genres'], axis=1)

    

    return df

        

data_train_cleaned = genres_FE(data_train_cleaned, genres_list_counter)

data_test_cleaned = genres_FE(data_test_cleaned, genres_list_counter)
display(data_train_cleaned.head(n=1))
# Lets get the list of production companies

prod_companies_list = flatten_data_column(data_train_cleaned['production_companies'])

prod_companies_list_test = flatten_data_column(data_test_cleaned['production_companies'])
# Create a counter of the most popular

prod_companies_list_counter = create_counter(prod_companies_list)

prod_companies_list_counter_test = create_counter(prod_companies_list_test)
# Let's select the top 30 companies

print(prod_companies_list_counter[:30])
def prod_companies_FE(df, prod_companies_counter, limit=100):

    df['num_production_companies'] = df['production_companies'].apply(lambda x: len(x) if x != {} else 0)

    df['all_production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x])

    

    for prod_company, count in prod_companies_counter[:limit]:

        df['production_company_' + "_".join(prod_company.split(" "))] = df['all_production_companies'].apply(lambda pcomp_list: 1 if prod_company in pcomp_list else 0)

    

    df['all_production_companies'] = df['all_production_companies'].apply(lambda x: " ".join(x))

    df = df.drop(['production_companies'], axis=1)

        

    return df



data_train_cleaned = prod_companies_FE(data_train_cleaned, prod_companies_list_counter, 30)

data_test_cleaned = prod_companies_FE(data_test_cleaned, prod_companies_list_counter_test, 30)
# Lets get the list of spoken languages

spoken_languages_list = flatten_data_column(data_train_cleaned['spoken_languages'])

spoken_languages_list_test = flatten_data_column(data_test_cleaned['spoken_languages'])



# Create a counter of the most popular

spoken_languages_list_counter = create_counter(spoken_languages_list)

spoken_languages_list_counter_test = create_counter(spoken_languages_list_test)
print(spoken_languages_list_counter)
def spoken_languages_FE(df, spoken_languages_counter, limit=56):

    df['num_spoken_languages'] = df['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

    df['all_spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x])

    

    for spoken_language, count in spoken_languages_counter[:limit]:

        df['language_' + spoken_language] = df['all_spoken_languages'].apply(lambda language_list: 1 if spoken_language in language_list else 0)

    

    df['all_spoken_languages'] = df['all_spoken_languages'].apply(lambda x: " ".join(x))

    df = df.drop(['spoken_languages'], axis=1)

        

    return df



spoken_languages_limit = 25



data_train_cleaned = spoken_languages_FE(data_train_cleaned, spoken_languages_list_counter, spoken_languages_limit)

data_test_cleaned = spoken_languages_FE(data_test_cleaned, spoken_languages_list_counter_test, spoken_languages_limit)



display(data_train_cleaned.head(n=2))
# Creates a List of orignal languages

def list_original_languages(mylist):

    flattened_list = []

    for element in mylist:

            flattened_list.append(element)

    return flattened_list

# Lets get the list of spoken languages

original_language_list = list_original_languages(data_train_cleaned['original_language'])

original_language_list_test = list_original_languages(data_test_cleaned['original_language'])



# Create a counter of the most popular

original_language_list_counter = Counter(original_language_list).most_common()

original_language_list_counter_test = Counter(original_language_list_test).most_common()



print(original_language_list_counter)

print(len(original_language_list_counter))
def original_language_FE(df, original_languages_counter, limit=56):

    for original_language, count in original_languages_counter:

        df['original_language_' + original_language] = df['original_language'].apply(lambda lang: 1 if lang == original_language else 0)



    df = df.drop(['original_language'], axis=1)

        

    return df



data_train_cleaned = original_language_FE(data_train_cleaned, original_language_list_counter)

data_test_cleaned = original_language_FE(data_test_cleaned, original_language_list_counter_test)



display(data_train_cleaned.head(n=2))
# Lets get the list of kewords

keywords_list = flatten_data_column(data_train_cleaned['Keywords'])

keywords_list_test = flatten_data_column(data_test_cleaned['Keywords'])



# Create a counter of the most popular keywords

keywords_list_counter = create_counter(keywords_list)

keywords_list_counter_test = create_counter(keywords_list_test)
print(keywords_list_counter[:30])
def keywords_FE(df, keywords_counter, limit=100):

    df['num_keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)

    df['all_keywords'] = df['Keywords'].apply(lambda x: [i['name'] for i in x])

    

    for keyword, count in keywords_counter[:limit]:

        df['keyword_' + keyword] = df['all_keywords'].apply(lambda keyword_list: 1 if keyword in keyword_list else 0)

    

    df['all_keywords'] = df['all_keywords'].apply(lambda x: " ".join(x))

    df = df.drop(['Keywords'], axis=1)

        

    return df



keywords_limit = 30



data_train_cleaned = keywords_FE(data_train_cleaned, keywords_list_counter, keywords_limit)

data_test_cleaned = keywords_FE(data_test_cleaned, keywords_list_counter_test, keywords_limit)



display(data_train_cleaned.head(n=2))
# Lets get the list of spoken languages

status_list = list_original_languages(data_train_cleaned['status'])

status_list_test = list_original_languages(data_test_cleaned['status'])



# Create a counter of the most popular

status_list_counter = Counter(status_list).most_common()

status_list_counter_test = Counter(status_list_test).most_common()



print(status_list_counter)

print(status_list_counter_test)
def status_FE(df):

    # Status: 1 released, 0 not released.

    df['status_released'] = df['status'].apply(lambda x: 1 if x == 'Released' else 0)

    df = df.drop(['status'], axis=1)

    

    return df
'''

data_train_cleaned = status_FE(data_train_cleaned)

data_test_cleaned = status_FE(data_test_cleaned)



display(data_train_cleaned.head(n=2))

'''
# Lets get the list of production countries

prod_countries_list = flatten_data_column(data_train_cleaned['production_countries'])

prod_countries_list_test = flatten_data_column(data_test_cleaned['production_countries'])



# Create a counter of the most popular production countries

prod_countries_list_counter = create_counter(prod_countries_list)

prod_countries_list_counter_test = create_counter(prod_countries_list_test)
def prod_countries_FE(df, prod_countries_counter, limit=100):

    df['num_production_countries'] = df['production_countries'].apply(lambda x: len(x) if x != {} else 0)

    df['all_production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x])

    

    for prod_country, count in prod_countries_counter[:limit]:

        df['production_country_' + "_".join(prod_country.split(" "))] = df['all_production_countries'].apply(lambda pcountry_list: 1 if prod_country in pcountry_list else 0)

    

    df['all_production_countries'] = df['all_production_countries'].apply(lambda x: " ".join(x))    

    df = df.drop(['production_countries'], axis=1)

        

    return df



prod_countries_limit = 30



data_train_cleaned = prod_countries_FE(data_train_cleaned, prod_countries_list_counter, prod_countries_limit)

data_test_cleaned = prod_countries_FE(data_test_cleaned, prod_countries_list_counter_test, prod_countries_limit)



display(data_train_cleaned.head(n=2))
display(data_train_cleaned.iloc[0]['cast'])
# Lets get the list of different cast

cast_list = flatten_data_column(data_train_cleaned['cast'])

cast_list_test = flatten_data_column(data_test_cleaned['cast'])



cast_name_list_counter = create_counter(cast_list, 'name')

cast_name_list_counter_test = create_counter(cast_list_test, 'name')



cast_character_list_counter = create_counter(cast_list, 'character')

cast_character_list_counter_test = create_counter(cast_list_test, 'character')





print(cast_name_list_counter[:10])

print(cast_character_list_counter[:10])
def cast_FE(df, cast_counter, character_counter, limit=100):

    df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    df['all_cast'] = df['cast'].apply(lambda x: [i['name'] for i in x])

    df['all_characters'] = df['cast'].apply(lambda x: [i['character'] for i in x])

    

    # Get the sum of each of the cast genders in a film: 0 `unknown`, 1 `female`, 2 `male`

    df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    

    # Create new columns for actors

    for cast, count in cast_counter[:limit]:

        df['cast_name_' + "_".join(cast.split(" "))] = df['all_cast'].apply(lambda cast_list: 1 if cast in cast_list else 0)

        

    # Create new columns for characters

    for character, count in character_counter[:limit]:

        df['cast_char_' + "_".join(character.split(" "))] = df['all_characters'].apply(lambda char_list: 1 if character in char_list else 0)

    

    df['all_cast'] = df['all_cast'].apply(lambda x: " ".join(x))

    df['all_characters'] = df['all_characters'].apply(lambda x: " ".join(x))  

    df = df.drop(['cast'], axis=1)

        

    return df



cast_limit = 30



data_train_cleaned = cast_FE(data_train_cleaned, cast_name_list_counter, cast_character_list_counter, cast_limit)

data_test_cleaned = cast_FE(data_test_cleaned, cast_name_list_counter_test, cast_character_list_counter_test, cast_limit)





display(data_train_cleaned.head(n=2))
display(data_train_cleaned.iloc[0]['crew'][:2])
# Lets get the list of different cast

crew_list = flatten_data_column(data_train_cleaned['crew'])

crew_list_test = flatten_data_column(data_test_cleaned['crew'])



crew_name_list_counter = create_counter(crew_list, 'name')

crew_name_list_counter_test = create_counter(crew_list_test, 'name')



crew_job_list_counter = create_counter(crew_list, 'job')

crew_job_list_counter_test = create_counter(crew_list_test, 'job')



crew_dep_list_counter = create_counter(crew_list, 'department')

crew_dep_list_counter_test = create_counter(crew_list_test, 'department')



print(crew_name_list_counter[:10])

print(crew_job_list_counter[:10])

print(crew_dep_list_counter[:10])
def crew_FE(df, crew_counter, job_counter, dep_counter, limit=100):

    df['num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

    df['all_crew'] = df['crew'].apply(lambda x: [i['name'] for i in x])



    

    # Get the sum of each of the cast genders in a film: 0 `unknown`, 1 `female`, 2 `male`

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    

    # Create new columns for crew members

    for crew_name, count in crew_counter[:limit]:

        df['crew_name_' + "_".join(crew_name.split(" "))] = df['all_crew'].apply(lambda crew_list: 1 if crew_name in crew_list else 0)

        

    # Create new columns for crew jobs

    for crew_job, count in job_counter[:limit]:

        df['crew_job_' + "_".join(crew_job.split(" "))] = df['crew'].apply(lambda job_list: sum([1 for i in job_list if i['job'] == crew_job]))

    

    # Create new columns for crew deparments

    for crew_dep, count in job_counter[:limit]:

        df['crew_department_' + "_".join(crew_dep.split(" "))] = df['crew'].apply(lambda dep_list: sum([1 for i in dep_list if i['job'] == crew_dep]))

        

    

    df['all_crew'] = df['all_crew'].apply(lambda x: " ".join(x))  

    df = df.drop(['crew'], axis=1)

        

    return df



crew_limit = 30



data_train_cleaned = crew_FE(data_train_cleaned, crew_name_list_counter, crew_job_list_counter, crew_dep_list_counter, crew_limit)

data_test_cleaned = crew_FE(data_test_cleaned, crew_name_list_counter_test, crew_job_list_counter_test, crew_dep_list_counter_test, crew_limit)



display(data_train_cleaned.head(n=2))
# Formula to apply logarithmic transformation for skewed data

def log_transform(df, feature):

    df['log_' + feature] = df[feature].apply(lambda x: np.log(x + 1))

    return df
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,10))

fig.suptitle('Budget Distribution', fontsize=15)

fig.subplots_adjust(hspace=0.5)



ax[0,0].set_title("Train data")

plt1 = sns.distplot(data_train_cleaned['budget'], ax=ax[0,0])

plt1.grid()



ax[0,1].set_title("Test data")

plt2 = sns.distplot(data_test_cleaned['budget'], ax=ax[0,1])

plt2.grid()



ax[1,0].set_title("Train data log")

plt1 = sns.distplot(log_transform(data_train_cleaned, 'budget')['log_budget'], ax=ax[1,0])

plt1.grid()



ax[1,1].set_title("Test data log")

plt2 = sns.distplot(log_transform(data_test_cleaned, 'budget')['log_budget'], ax=ax[1,1])

plt2.grid()
# Calculate skweness

print("Skweness budget for train data: {}".format(stats.skew(data_train_cleaned['budget'])))

print("Skweness budget for test data: {}\n".format(stats.skew(data_test_cleaned['budget'])))



# Skeweness of log data

print("Skweness budget for log train data: {}".format(stats.skew(log_transform(data_train_cleaned, 'budget')['log_budget'])))

print("Skweness budget for log test data: {}".format(stats.skew(log_transform(data_test_cleaned, 'budget')['log_budget'])))

data_train_cleaned = data_train_cleaned.drop(['budget'], axis=1)

data_test_cleaned = data_test_cleaned.drop(['budget'], axis=1)
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,10))

fig.suptitle('Popularity Distribution', fontsize=15)

fig.subplots_adjust(hspace=0.5)





ax[0,0].set_title("Train data")

plt1 = sns.distplot(data_train_cleaned['popularity'], bins=50, ax=ax[0,0])

plt1.grid()



ax[0,1].set_title("Test data")

plt2 = sns.distplot(data_test_cleaned['popularity'],  bins=50, ax=ax[0,1])

plt2.grid()





ax[1,0].set_title("Train data log")

plt1 = sns.distplot(log_transform(data_train_cleaned, 'popularity')['log_popularity'], ax=ax[1,0])

plt1.grid()



ax[1,1].set_title("Test data log")

plt2 = sns.distplot(log_transform(data_test_cleaned, 'popularity')['log_popularity'], ax=ax[1,1])

plt2.grid()
print("Skweness popularity for train data: {}".format(stats.skew(data_train_cleaned['popularity'])))

print("Skweness popularity for test data: {}\n".format(stats.skew(data_test_cleaned['popularity'])))



print("Skweness popularity for log train data: {}".format(stats.skew(data_train_cleaned['log_popularity'])))

print("Skweness popularity for log test data: {}".format(stats.skew(data_test_cleaned['log_popularity'])))
data_train_cleaned = data_train_cleaned.drop(['popularity'], axis=1)

data_test_cleaned = data_test_cleaned.drop(['popularity'], axis=1)
fig, ax = plt.subplots(ncols=2, figsize=(20,5))

fig.suptitle('Revenue Distribution', fontsize=15)

fig.subplots_adjust(hspace=0.5)





ax[0].set_title("Train data")

plt1 = sns.distplot(data_train_cleaned['revenue'], bins=50, ax=ax[0])

plt1.grid()



ax[1].set_title("Train data log")

plt1 = sns.distplot(log_transform(data_train_cleaned, 'revenue')['log_revenue'], ax=ax[1])

plt1.grid()

print("Skweness popularity for train data: {}".format(stats.skew(data_train_cleaned['revenue'])))

print("Skweness popularity for log train data: {}".format(stats.skew(data_train_cleaned['log_revenue'])))
# Drop the original values of revenue

data_train_cleaned = data_train_cleaned.drop(['revenue'], axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LinearRegression

import eli5



vectorizer_overview = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



vectorizer_overview = vectorizer_overview.fit(data_train_cleaned['overview'].fillna(''))

overview_text = vectorizer_overview.transform(data_train_cleaned['overview'].fillna(''))

linreg_overview = LinearRegression()

linreg_overview.fit(overview_text, data_train_cleaned['log_revenue'])

eli5.show_weights(linreg_overview, vec=vectorizer_overview, top=20, feature_filter=lambda x: x != '<BIAS>')
def overview_FE(df, predictor, vect):

    df['word_revenue'] = df['overview'].apply(lambda x: predictor.predict(vect.transform([x]))[0]) 

    

    return df



'''

data_train_cleaned =  overview_FE(data_train_cleaned, linreg, vectorizer_overview)



display(data_train_cleaned['word_revenue'].head(n=5))

'''
vectorizer_tagline = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



vectorizer_tagline = vectorizer_tagline.fit(data_train_cleaned['tagline'].fillna(''))

overview_text = vectorizer_tagline.transform(data_train_cleaned['tagline'].fillna(''))

linreg_tagline = LinearRegression()

linreg_tagline.fit(overview_text, data_train_cleaned['log_revenue'])

eli5.show_weights(linreg_tagline, vec=vectorizer_tagline, top=20, feature_filter=lambda x: x != '<BIAS>')
'''

def tagline_FE(df, predictor, vect):

    df['tagline_revenue'] = df['tagline'].apply(lambda x: predictor.predict(vect.transform([x]))[0]) 

    

    return df



data_train_cleaned =  tagline_FE(data_train_cleaned, linreg_tagline, vectorizer_tagline)



display(data_train_cleaned['tagline_revenue'].head(n=5))

'''
display(data_train_cleaned.head(n=2))
features_to_drop = ['original_title', 'overview', 'release_date', 'tagline', 'title',

                   'all_genres', 'all_production_companies', 'all_spoken_languages',

                   'all_keywords', 'all_production_countries', 'all_cast', 'all_cast',

                   'all_crew', 'all_characters']



def drop_features(df, features_list):

    df = df.drop(features_list, axis=1)

    

    return df
data_train_cleaned = drop_features(data_train_cleaned, features_to_drop)

data_test_cleaned = drop_features(data_test_cleaned, features_to_drop)



display(data_train_cleaned.head())

display(data_test_cleaned.head())
def feature_engineering (df):

    """

    Data Pre-Processing

    """

    df = runtime_pre_process(df)

    

    df = release_year_pre_process(df)

    

    df = homepage_pre_process(df)

    

    df = poster_pre_process(df)

    

    df = imputing_budget(df)

    

    df = collection_pre_process(df)

    

    df = genres_pre_processing(df)

    

    df = imput_title(df)

    

    df = prod_comp_pre_processing(df)

    

    df = spoken_lang_pre_processing(df)

    

    df = keywords_pre_processing(df)

    

    df = cast_pre_processing(df)

    

    df = crew_pre_processing(df)

    

    df = overview_pre_process(df)

    

    df = tagline_pre_process(df)

    

    df = prod_countries_pre_process(df)

    

    df = status_pre_process(df)

    

    """

    Feature Engineering

    """

    # Convert panda string to list

    df = string2_list(df)

    

    # Genres

    genres_list = flatten_data_column(data_train_preprocessed['genres'])

    genres_list_counter = create_counter(genres_list)

    df = genres_FE(df, genres_list_counter)

    

    # Production Companies

    prod_companies_list = flatten_data_column(data_train_preprocessed['production_companies'])

    prod_companies_list_counter = create_counter(prod_companies_list)

    df = prod_companies_FE(df, prod_companies_list_counter, 30)

    

    # Spoken Languages

    spoken_languages_list = flatten_data_column(data_train_preprocessed['spoken_languages'])

    spoken_languages_list_counter = create_counter(spoken_languages_list)

    df = spoken_languages_FE(df, spoken_languages_list_counter, 25)

    

    # Original Language

    original_language_list = list_original_languages(data_train_preprocessed['original_language'])

    original_language_list_counter = Counter(original_language_list).most_common()

    df = original_language_FE(df, original_language_list_counter)

    

    # Keywords

    keywords_list = flatten_data_column(data_train_preprocessed['Keywords'])

    keywords_list_counter = create_counter(keywords_list)

    df = keywords_FE(df, keywords_list_counter, 30)

    

    # Status

    status_list = list_original_languages(data_train_preprocessed['status'])

    status_list_counter = Counter(status_list).most_common()

    df = status_FE(df)

    

    # Production Countries

    prod_countries_list = flatten_data_column(data_train_preprocessed['production_countries'])

    prod_countries_list_counter = create_counter(prod_countries_list)

    df = prod_countries_FE(df, prod_countries_list_counter, 30)

    

    # Cast

    cast_list = flatten_data_column(data_train_preprocessed['cast'])

    cast_name_list_counter = create_counter(cast_list, 'name')

    cast_character_list_counter = create_counter(cast_list, 'character')

    df = cast_FE(df, cast_name_list_counter, cast_character_list_counter, 30)

    

    # Crew

    crew_list = flatten_data_column(data_train_preprocessed['crew'])

    crew_name_list_counter = create_counter(crew_list, 'name')

    crew_job_list_counter = create_counter(crew_list, 'job')

    crew_dep_list_counter = create_counter(crew_list, 'department')

    df = crew_FE(df, crew_name_list_counter, crew_job_list_counter, crew_dep_list_counter, 30)

    

    # Budget fix skewness

    df = log_transform(df, 'budget')

    df = df.drop(['budget'], axis=1)

    

    # Popularity fix skewness

    df = log_transform(df, 'popularity')

    df = df.drop(['popularity'], axis=1)



    

    # Revenue fix skewness

    #df = log_transform(df, 'revenue')

    #df = df.drop(['revenue'], axis=1)

    

    # Overview ---> Overfits the data

    # df =  overview_FE(df, linreg_overview, vectorizer_overview)

    

    # Tagline ----> Overfits the data

    # df =  tagline_FE(df, linreg_tagline, vectorizer_tagline)



    

    # Drop unwanted Features

    features_to_drop = ['original_title', 'overview', 'release_date', 'tagline', 'title',

                   'all_genres', 'all_production_companies', 'all_spoken_languages',

                   'all_keywords', 'all_production_countries', 'all_cast', 'all_cast',

                   'all_crew', 'all_characters', 'imdb_id']

    

    df = drop_features(df, features_to_drop)

    

    return df
from sklearn.model_selection import train_test_split



seed = np.random.seed(34)



# Train data

data_train_bench = data_train_raw.copy().drop(['id', 'revenue'], axis=1)

data_train_bench = feature_engineering(data_train_bench)

# Test data

data_test_bench = data_test_raw.copy().drop(['id'], axis=1)

data_test_bench = feature_engineering(data_test_bench)



# Target data

target_train_bench = data_train_cleaned['log_revenue']



train_X_bench,test_X_bench, train_y_bench, test_y_bench = train_test_split(data_train_bench,

                                                                           target_train_bench,

                                                                           test_size=0.25,

                                                                           random_state=seed)
train_data_bench = lgb.Dataset(train_X_bench, label=train_y_bench)

val_data_bench = lgb.Dataset(test_X_bench, label=test_y_bench)



num_round = 1000



param = {'num_leaves':31,

         'metric': 'rmse',

         'objective':'regression'}



# Train the model

clf = lgb.train(param,

                train_data_bench,

                num_round,

                valid_sets=[train_data_bench, val_data_bench],

                verbose_eval=5000,

                early_stopping_rounds=100)



ypred = clf.predict(test_X_bench, num_iteration=clf.best_iteration)
# Calculate the rms of the model.

rms_benchmark = sqrt(mean_squared_error(test_y_bench, ypred))

print("The rms score is: {}".format(rms_benchmark))
# Separate the target variable 

target = data_train_cleaned['log_revenue']

# Drop target data from train

data_train = data_train_raw.copy().drop(['id', 'revenue'], axis=1)



# Test data

data_test = data_test_raw.copy().drop(['id'], axis=1)

data_test = feature_engineering(data_test)
from sklearn.model_selection import KFold



# Split k-fold validation

n_splits = 7

random_state = np.random.seed(654658)



kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 3,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}
# oof -> Out of fold. One single vector with all the validation predictions to

# then calculate the error upon this predictions.

oof = np.zeros(len(data_train))

predictions_test = np.zeros(len(data_test_cleaned))



# K-fold CV

for epoch, (train_index, val_index) in enumerate(kf.split(data_train.values)):

    # Only temporary

    training_data = data_train.copy()

    

    X_train, X_val = training_data.loc[train_index], training_data.loc[val_index]

    y_train, y_val = target.loc[train_index], target.loc[val_index]

    

    # Need to fix this

    X_train = feature_engineering(X_train)

    X_val = feature_engineering(X_val)

    

    print("Fold index: {}".format(epoch + 1))

    

    train_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_val, label=y_val)

    

    num_round = 1000000

    

    # Train the model

    clf = lgb.train(params,

                    train_data,

                    num_round,

                    valid_sets=[train_data, val_data],

                    verbose_eval=5000,

                    early_stopping_rounds=1000) 

    

    # Out of fold vector

    oof[val_index] = clf.predict(X_val.astype('float32'), num_iteration=clf.best_iteration)

    

    # Calculate the average predictions for all folds for the test submsission data

    predictions_test += (clf.predict(data_test, num_iteration=clf.best_iteration) / kf.n_splits)
# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, oof))

print("The rms score is: {}".format(rms))
from catboost import Pool, CatBoostRegressor



clf_cat = CatBoostRegressor(iterations=10000,

                            learning_rate=0.01,

                            depth=5, 

                            eval_metric='RMSE',

                            random_seed=23,

                            early_stopping_rounds=200,

                            logging_level='Verbose')
# oof -> Out of fold. One single vector with all the validation predictions to

# then calculate the error upon this predictions.

oof_cat = np.zeros(len(data_train))

predictions_test_cat = np.zeros(len(data_test_cleaned))



# K-fold CV

for epoch, (train_index, val_index) in enumerate(kf.split(data_train.values)):

    training_data = data_train.copy()

    

    X_train, X_val = training_data.loc[train_index], training_data.loc[val_index]

    y_train, y_val = target.loc[train_index], target.loc[val_index]

    

    X_train = feature_engineering(X_train)

    X_val = feature_engineering(X_val)

    

    print("Fold index: {}".format(epoch + 1))

    

    # train the model

    clf_cat.fit(X_train, y_train,

                eval_set=(X_val,y_val),

                use_best_model=True,

                verbose=False)

    

    # Out of fold vector

    oof_cat[val_index] = clf_cat.predict(X_val.astype('float32'))

    

    # Calculate the average predictions for all folds for the test submsission data

    predictions_test_cat += (clf_cat.predict(data_test) / kf.n_splits)
# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, oof_cat))

print("The rms score is: {}".format(rms))
# Imports

import torch

from torch import nn

import torch.nn.functional as F

from torch import optim

import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler







def scaled_data(df):

    scaler = MinMaxScaler()

    features = [feature for feature in df.columns]

    

    df[features] = scaler.fit_transform(df[features])

    return df
# Checks if GPU is available.

isGPUAvailable = torch.cuda.is_available()

device = "cpu"



if isGPUAvailable:

    device = "cuda"

    print("Training on GPU")

else:

    device = "cpu"

    print("Training on CPU")
# Model

class Classifier(nn.Module):

    def __init__(self, in_classes, dropout=0.5):

        super().__init__()

        self.input_dim = in_classes

        self.hidden_1 = int(self.input_dim)

        self.fc1 = nn.Linear(self.input_dim, int(self.hidden_1))

        self.fc2 = nn.Linear(int(self.hidden_1), int(self.hidden_1/2))

        self.fc3 = nn.Linear(int(self.hidden_1/2), 1)

        #self.fc4 = nn.Linear(int(self.hidden_1/2), int(self.hidden_1/4))

        #self.fc5 = nn.Linear(int(self.hidden_1/4), 1)



        # Dropout

        self.dropout = nn.Dropout(p=dropout)



    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        #x = self.dropout(F.relu(self.fc3(x)))

        #x = self.dropout(F.relu(self.fc4(x)))





        # output so no dropout here

        out = self.fc3(x)



        return out
## Create the model instance

in_classes = data_test.shape[1]

dropout = 0.4





# Create the feed forward deep learning classifier

ff_classifier = Classifier(in_classes=in_classes, dropout=dropout)



# Move classifier to GPU

ff_classifier.to(device)
# Train and validation

def train (clf, train_loader, valid_loader, epochs=40, min_valid_loss=np.Inf, lr=0.01):

    best_clf = clf

    optimizer = torch.optim.SGD(clf.parameters(), lr = lr, momentum=0.5, nesterov=True)

    #optimizer = torch.optim.Adam(clf.parameters(), lr = lr, weight_decay=0.0001)

    #optimizer = torch.optim.RMSprop(clf.parameters(), lr = lr, weight_decay=0.0001, momentum=0.1)

    

    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True,

                                                     patience=20, min_lr=0.00001)

    

    for epoch in range(epochs):

        train_loss = 0.0

        validation_loss = 0.0

        

        

        # train

        clf.train()

        for inputs, labels in train_loader:

            # Reset optimizer for every iteration.

            optimizer.zero_grad()

            

            # Move tensors to GPU

            inputs, labels = inputs.to(device), labels.to(device)

            

            # Forward pass

            output = clf(inputs)

            

            # Loss

            loss = criterion(output.squeeze(), labels)

            

            # Backward pass (Backpropagation)

            loss.backward()

            

            # Update weights

            optimizer.step()

            

            # Update the loss.

            train_loss += loss.item()

            

        # Validation

        clf.eval()

        for inputs, labels in valid_loader:

            # Move tensors to GPU

            inputs, labels = inputs.to(device), labels.to(device)

            

            # Forward pass

            output = clf(inputs)

            

            # Loss

            loss = criterion(output.squeeze(), labels)

            

            # Update the validation loss.

            validation_loss += loss.item()

            

        

        # Calculate the losses.

        train_loss = train_loss/len(train_loader)

        validation_loss = validation_loss/len(valid_loader)

        

        #Update lr

        scheduler.step(validation_loss)

        

        # Print the losses

        print("Epoch {0}".format(epoch + 1))

        #print('LR:', scheduler.get_lr())

        print("Train loss = {0}".format(train_loss))

        print("Validation loss = {0}".format(validation_loss))

        

        # Check if validation loss has reduced, and therefore the model predicts better

        if validation_loss < min_valid_loss:

            min_valid_loss = validation_loss

            print("Validation loss has decreased. Saving the model...")

            best_clf = clf

        print("------------------------------------")

    return best_clf
# K-fold cross validation

lr = 0.001

epochs = 40

batch_size = 60



# Convert test data to tensor

x_test = np.array(scaled_data(data_test))

x_test_tensor = torch.tensor(x_test, dtype=torch.float).to(device)



test_preds_nn = np.zeros((len(data_test)))

oof_nn = np.zeros(len(data_train))





for fold_i, (train_index, val_index) in enumerate(kf.split(data_train.values)):

    

    print("\n")

    print("Fold {0}".format(fold_i + 1))

    

    training_data = data_train.copy()

    

    x_train_raw, x_val_raw = training_data.loc[train_index], training_data.loc[val_index]

    y_train_raw, y_val_raw = target.loc[train_index].values, target.loc[val_index].values

    

    x_train_raw = np.array(scaled_data(feature_engineering(x_train_raw)))

    x_val_raw = np.array(scaled_data(feature_engineering(x_val_raw)))

    

    

    x_train_fold = torch.tensor(x_train_raw, dtype=torch.float)

    y_train_fold = torch.tensor(y_train_raw, dtype=torch.float32)

    

    x_val_fold = torch.tensor(x_val_raw, dtype=torch.float)

    y_val_fold = torch.tensor(y_val_raw, dtype=torch.float32)

    

    train_dataset = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

    valid_dataset = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    

    

    model = train(ff_classifier, train_loader, valid_loader, epochs=epochs, lr = lr)

    

    # Out of fold data to check accuracy at the end

    oof_nn[val_index] = model(x_val_fold.to(device)).squeeze().to('cpu').detach().numpy()   

    

    test_preds_nn += model(x_test_tensor).squeeze().to('cpu').detach().numpy() / kf.n_splits
# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, oof_nn))

print("The rms score is: {}".format(rms))
ensemble_oof = 0.3*oof + 0.4*oof_nn + 0.3*oof_cat
# Calculate the rms of the model.

rms_lgbm = sqrt(mean_squared_error(target, oof))

print("The rms score for LigthGBM is: {}".format(rms_lgbm))



# Calculate the rms of the model.

rms_lgbm = sqrt(mean_squared_error(target, oof_nn))

print("The rms score for Neural Netowork is: {}".format(rms_lgbm))



# Calculate the rms of the model.

rms_lgbm = sqrt(mean_squared_error(target, oof_cat))

print("The rms score for CatBoost is: {}".format(rms_lgbm))



# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, ensemble_oof))

print("The rms score for ensemble is: {}".format(rms))
ensemble_oof_2 = 0.4*oof + 0.3*oof_nn + 0.3*oof_cat

ensemble_oof_3 = 0.3*oof + 0.3*oof_nn + 0.4*oof_cat

ensemble_trees = 0.6*oof + 0.4*oof_cat





# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, ensemble_oof_2))

print("The rms score for ensemble 2 is: {}".format(rms))

# Calculate the rms of the model.

rms = sqrt(mean_squared_error(target, ensemble_oof_3))

print("The rms score for ensemble 3 is: {}".format(rms))
# Log PREDICTION results from the lgbm

log_predictions_lgbm = pd.DataFrame(data=predictions_test, columns=['revenue'])

# Undo logs for the final submission

predictions_lgbm = log_predictions_lgbm['revenue'].apply(lambda x: np.exp(x) - 1)
# Log PREDICTION results from the Neural Network

log_predictions_nn = pd.DataFrame(data=test_preds_nn, columns=['revenue'])

# Undo logs for the final submission

predictions_nn = log_predictions_nn['revenue'].apply(lambda x: np.exp(x) - 1)
# Log PREDICTION results from the Neural Network

log_predictions_cat = pd.DataFrame(data=predictions_test_cat, columns=['revenue'])

# Undo logs for the final submission

predictions_cat = log_predictions_nn['revenue'].apply(lambda x: np.exp(x) - 1)
ensemble_submission = 0.3*predictions_lgbm + 0.4*predictions_nn + 0.3*predictions_cat

ensemble_submission_2 = 0.4*predictions_lgbm + 0.3*predictions_nn + 0.3*predictions_cat

ensemble_submission_3 = 0.3*predictions_lgbm + 0.3*predictions_nn + 0.4*predictions_cat
display(ensemble_submission.head())
# Submission 1

submission_ensemble = pd.DataFrame({"id": data_test_raw["id"].values})

submission_ensemble["revenue"] = ensemble_submission

submission_ensemble.to_csv("submission.csv", index=False)



# Submission 2

submission_ensemble_2 = pd.DataFrame({"id": data_test_raw["id"].values})

submission_ensemble_2["revenue"] = ensemble_submission_2

submission_ensemble_2.to_csv("submission2.csv", index=False)



# Submission 3

submission_ensemble_3 = pd.DataFrame({"id": data_test_raw["id"].values})

submission_ensemble_3["revenue"] = ensemble_submission_3

submission_ensemble_3.to_csv("submission3.csv", index=False)
display(len(submission_ensemble))